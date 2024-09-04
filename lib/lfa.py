from pandapower.plotting import plotly
from datetime import datetime
from typing import List, Any
import pandapower as pp
import os, json, uuid
import polars as pl
import time

from lib.schemas.topology import (
    Topology,
    UsagePoint,
    ConnectivityNode,
    PowerTransformer,
    AcLineSegment,
    Switch,
    GhostNodes
)

from lib import logger, decorator_timer

PATH = os.path.dirname(os.path.abspath(__file__))


def read_topology_data(topology_path: str):
    with open(os.path.join(topology_path), 'r') as fp:
        data = json.load(fp)
        return Topology(**data)


class DataLoader():
    def __init__(self, data_path: str, work_path: str, loads: List[UsagePoint]):
        df = pl.DataFrame()
        self.data_path = os.path.join(work_path, 'data.parquet')
        if not os.path.exists(self.data_path):
            for i, load in enumerate(loads):
                if not os.path.exists(os.path.join(data_path, load)):
                    logger.warning(f"[{i+1} of {len(loads)}] cannot load ami data for meter {load}")
                else:
                    df = df.vstack(pl.read_parquet(os.path.join(data_path, load)))
            (df.with_columns(((pl.col('p_kwh_in') - pl.col('p_kwh_out')) / 1e3).alias('p_mw'),
                             ((pl.col('q_kvarh_in') - pl.col('q_kvarh_out')) / 1e3).alias('q_mvar'))
             .with_columns(((pl.col('p_mw') ** 2 + pl.col('q_mvar') ** 2) ** 0.5).alias('s_mva'), )
             .drop('p_kwh_in', 'p_kwh_out', 'q_kvarh_in', 'q_kvarh_out')
             .write_parquet(self.data_path))
            logger.info(f"Prepared data for {df.n_unique('meter_id')} of {len(loads)} ({len(loads)-df.n_unique('meter_id')} missing)")

    def load_profile_iter(self, from_date: datetime, to_date: datetime = None):
        if to_date is None:
            to_date = from_date

        df = pl.read_parquet(self.data_path).filter(pl.col('datetime').is_between(from_date, to_date))
        # pl.read_parquet(self.data_path).group_by('meter_id').agg(peak_s_mva=pl.col('s_mva').max()).sort(by='peak_s_mva', descending=True)
        for batch in df.partition_by('datetime'):
            yield batch


class Lfa(DataLoader):
    def __init__(
            self,
            work_path: str,
            data_path: str,
            lv_path: str,
            mv_path: str,
    ):
        os.makedirs(work_path, exist_ok=True)
        self.work_path = work_path
        self.mv_path = mv_path
        self.lv_path = lv_path
        self.net = self.read_net()
        super().__init__(
            data_path=data_path,
            work_path=work_path,
            loads=list(self.net.load['name'])
        )

    @staticmethod
    def create_region(self, lv_topology_list: List[dict], mv_topology_list: List[dict]) -> pp.pandapowerNet:

        assert len(mv_topology_list) == 1, f'Multiple MV regional layers are not supported'

        net = pp.create_empty_network(name=mv_topology_list[0]['uuid'])

        def pp_bus(bus: str):
            try:
                net.bus.loc[net.bus['name'] == bus].index.item()
            except Exception as e:
                print(e)

        # add region bussed for mv and lv topologies
        for bus in mv_topology_list[0]['bus']:
            pp.create_bus(net, name=bus['bus'], type='n', vn_kv=bus['rated_kv'], in_service=bus['in_service'])
        for lv_topology in lv_topology_list:
            for bus in lv_topology['bus']:
                pp.create_bus(net, name=bus['bus'], type='n', vn_kv=bus['rated_kv'], in_service=bus['in_service'])

        # configure mv slack bus
        for bus in mv_topology_list[0]['slack']:
            pp.create_bus(net, name=bus['bus'], type='n', vn_kv=bus['rated_kv'], in_service=bus['in_service'])
            pp.create_ext_grid(net, name=bus['bus'], vm_pu=1.0, bus=pp_bus(bus=bus['bus']))

        print('done')

    @staticmethod
    def create_net(lv: List[Topology], mv: Topology) -> pp.pandapowerNet:

        logger.debug(msg=f'Creating subnet for mv topology: {mv.uuid}')

        net = pp.create_empty_network(name=mv.uuid)

        #
        # net element helper functions
        #
        def pp_bus(bus: str):
            try:
                return net.bus.loc[net.bus['name'] == bus].index.item()
            except Exception as e:
                raise Exception(f'pp_bus raised exception for multiple entries of bus={bus} detected net')

        def create_bus(bus: ConnectivityNode):
            if bus.bus not in net.bus['name']:
                pp.create_bus(
                    net,
                    name=bus.bus,
                    type='n',
                    vn_kv=bus.rated_kv,
                    in_service=bus.in_service
                )
            else:
                raise Exception(f'create_bus raised exception for multiple entries of bus={bus.bus} in net')

        def create_ext_grid(bus: ConnectivityNode):
            create_bus(bus=bus)
            if bus.bus not in net.ext_grid['name']:
                pp.create_ext_grid(
                    net,
                    name=bus.bus,
                    vm_pu=1.0,
                    bus=pp_bus(bus=bus.bus)
                )
            else:
                raise Exception(f'create_ext_grid raised exception for multiple external grid entries of bus={bus.bus} in net')

        def create_trafo(trafo: PowerTransformer):
            if trafo.name not in net.trafo['name']:
                pp.create_transformer_from_parameters(
                    net,
                    name=trafo.name,
                    mrid=trafo.mrid,
                    hv_bus=pp_bus(bus=trafo.hv_bus),
                    lv_bus=pp_bus(bus=trafo.lv_bus),
                    sn_mva=trafo.sn_mva,
                    vn_hv_kv=trafo.vn_hv_kv,
                    vn_lv_kv=trafo.vn_lv_kv,
                    in_service=trafo.in_service,
                    vkr_percent=5,  # TODO Verify number
                    vk_percent=10,  # TODO Verify number
                    pfe_kw=2,  # TODO Verify number
                    i0_percent=3  # TODO Verify number
                )
            else:
                raise Exception(f'create_trafo raised exception for multiple trafo entries of trafo={trafo.name} in net')

        def create_branch(branch: AcLineSegment):
            if branch.has_impedance:
                if branch.name not in net.line['name']:
                    pp.create_line_from_parameters(
                        net,
                        name=branch.name,
                        mrid=branch.mrid,
                        from_bus=pp_bus(bus=branch.from_bus),
                        to_bus=pp_bus(bus=branch.to_bus),
                        length_km=1,  # TODO Verify number
                        r_ohm_per_km=branch.r,
                        x_ohm_per_km=branch.x,
                        c_nf_per_km=200,  # TODO Verify number
                        max_i_ka=1,  # TODO Verify number
                        in_service=True
                    )
                else:
                    raise Exception(f'create_branch raised exception for multiple branch entries for branch={branch.name} in net')
            else:
                create_switch(
                    element=branch
                )

        def create_switch(element: Any):
            if type(element) in [Switch, AcLineSegment] and element.mrid not in net.switch['name']:
                pp.create_switch(
                    net,
                    name=element.name,
                    mrid=element.mrid,
                    bus=pp_bus(bus=element.from_bus),
                    element=pp_bus(bus=element.to_bus),
                    et='b',
                    closed=True if type(element) == AcLineSegment else not bool(element.is_open)
                )
            else:
                raise Exception(f'create_switch raised exception for multiple switch entries for switch={element.name} in net')

        def create_load(load: UsagePoint):
            if load.bus not in net.load['name']:
                pp.create_load(
                    net,
                    name=load.meter_id,
                    mrid=load.mrid,
                    bus=pp_bus(bus=load.bus),
                    p_mw=0.0
                )
            else:
                raise Exception(f'create_load raised exception for multiple load entries for load={load.bus} in net')

        def create_ghost(ghost: GhostNodes):
            switch = Switch(
                mrid=ghost.mrid,
                from_bus=ghost.from_bus,
                to_bus=ghost.to_bus,
                is_open=False,
                name=f'branch_{uuid.uuid4().__str__()}'
            )
            create_switch(element=switch)

        #
        # add net elements
        #

        # busses
        logger.debug(msg=f'compile mv busses for uuid {mv.uuid}')
        for mv_bus in mv.bus:
            create_bus(bus=mv_bus)

        for i, lv_i in enumerate(lv):
            logger.debug(msg=f'compile lv bus group {i + 1} for uuid {lv_i.uuid}')
            for lv_bus in lv_i.bus:
                create_bus(bus=lv_bus)

        # slack busses
        for i, mv_slack_bus in enumerate(mv.slack):
            logger.debug(msg=f'compile mv slack bus {i + 1} for bus {mv_slack_bus}')
            create_ext_grid(bus=mv_slack_bus)

        # trafo's
        for i, mv_trafo in enumerate(mv.trafo):
            logger.debug(msg=f'compile mv trafo {i + 1} with name: {mv_trafo.name}={mv_trafo.mrid} (in-service={mv_trafo.in_service})')
            if mv_trafo.in_service:
                create_trafo(trafo=mv_trafo)

        # branches
        for i, mv_branch in enumerate(mv.branch):
            logger.debug(msg=f'compile mv branch {i + 1} with name: {mv_branch.name}={mv_branch.mrid}')
            create_branch(branch=mv_branch)

        for i, lv_i in enumerate(lv):
            logger.debug(msg=f'compile lv branch group {i + 1} for uuid {lv_i.uuid}')
            for lv_branch in lv_i.branch:
                create_branch(branch=lv_branch)

        # switches
        for i, mv_switch in enumerate(mv.switch):
            logger.debug(msg=f'compile mv switch {i + 1} with name: {mv_switch.name}={mv_switch.mrid}')
            create_switch(element=mv_switch)

        for i, lv_i in enumerate(lv):
            logger.debug(msg=f'compile lv with group {i + 1} for uuid {lv_i.uuid}')
            for lv_switch in lv_i.switch:
                create_switch(element=lv_switch)

        # loads
        for i, mv_laod in mv.load:
            logger.debug(msg=f'compile mv load {i + 1} with name: {mv_laod.name}={mv_laod.mrid}')
            create_load(load=mv_laod)

        for i, lv_i in enumerate(lv):
            logger.debug(msg=f'compile lv load group {i + 1} for uuid {lv_i.uuid}')
            for lv_load in lv_i.load:
                create_load(load=lv_load)

        # ghost nodes
        for i, mv_ghost in mv.ghost:
            logger.debug(msg=f'compile mv ghost node {i + 1} with mrid: {mv_ghost.mrid}')
            create_ghost(ghost=mv_ghost)

        return net

    @staticmethod
    def create_subnet(lv: Topology) -> pp.pandapowerNet:

        logger.debug(msg=f'Creating subnet for lv topology: {lv.uuid}')

        net = pp.create_empty_network(name=lv.uuid)

        def pp_bus(bus: str):
            try:
                return net.bus.loc[net.bus['name'] == bus].index.item()
            except Exception as e:
                print(e)

        for bus in lv.bus:
            pp.create_bus(
                net,
                name=bus.bus,
                type='n',
                vn_kv=bus.rated_kv,
                in_service=bus.in_service
            )

        for bus in lv.slack:
            pp.create_bus(net, name=bus.bus, type='n', vn_kv=bus.rated_kv, in_service=bus.in_service)
            pp.create_ext_grid(net, name=bus.bus, vm_pu=1.0, bus=pp_bus(bus=bus.bus))

        for trafo in lv.trafo:
            pp.create_transformer_from_parameters(
                net,
                name=trafo.name,
                mrid=trafo.mrid,
                hv_bus=pp_bus(bus=trafo.hv_bus),
                lv_bus=pp_bus(bus=trafo.lv_bus),
                sn_mva=trafo.sn_mva,
                vn_hv_kv=trafo.vn_hv_kv,
                vn_lv_kv=trafo.vn_lv_kv,
                in_service=trafo.in_service,
                vkr_percent=5,  # TODO Verify number
                vk_percent=10,  # TODO Verify number
                pfe_kw=2,  # TODO Verify number
                i0_percent=3  # TODO Verify number
            )

        for branch in lv.branch:
            if branch.has_impedance:
                pp.create_line_from_parameters(
                    net,
                    name=branch.name,
                    mrid=branch.mrid,
                    from_bus=pp_bus(bus=branch.from_bus),
                    to_bus=pp_bus(bus=branch.to_bus),
                    length_km=1,  # TODO Verify number
                    r_ohm_per_km=branch.r,
                    x_ohm_per_km=branch.x,
                    c_nf_per_km=200,  # TODO Verify number
                    max_i_ka=1,  # TODO Verify number
                    in_service=True
                )
            else:
                pp.create_switch(
                    net,
                    name=branch.name,
                    mrid=branch.mrid,
                    bus=pp_bus(bus=branch.from_bus),
                    element=pp_bus(bus=branch.to_bus),
                    et='b',
                    closed=True
                )

        for switch in lv.switch:
            pp.create_switch(
                net,
                name=switch.name,
                mrid=switch.mrid,
                bus=pp_bus(bus=switch.from_bus),
                element=pp_bus(bus=switch.to_bus),
                et='b',
                closed=not bool(switch.is_open)
            )

        for load in lv.load:
            pp.create_load(
                net,
                name=load.name,
                bus=pp_bus(bus=load.bus),
                p_mw=0.0,
            )

        for ghost in lv.ghost:
            pp.create_switch(net,
                             name=ghost.mrid,
                             bus=pp_bus(bus=ghost.from_bus),
                             element=pp_bus(bus=ghost.to_bus),
                             et='b',
                             closed=False
                             )

        return net

    @decorator_timer
    @staticmethod
    def run_lfa(net: pp.pandapowerNet):
        pp.runpp(net)

    def write_net(self, net: pp.pandapowerNet):
        if os.path.exists(os.path.join(self.work_path, 'net.sqlite')):
            os.remove(os.path.join(self.work_path, 'net.sqlite'))
        pp.to_sqlite(net, os.path.join(self.work_path, 'net.sqlite'))

    def read_net(self) -> pp.pandapowerNet:
        if os.path.isfile(os.path.join(self.work_path, 'net.sqlite')):
            return pp.from_sqlite(os.path.join(self.work_path, 'net.sqlite'))

        def read(topology_path: str) -> Topology:
            with open(os.path.join(topology_path), 'r') as fp:
                return Topology(**json.load(fp))

        net = self.create_net(
            lv=[read(topology_path=os.path.join(self.lv_path, lv_name)) for lv_name in os.listdir(self.lv_path)],
            mv=read(topology_path=os.path.join(self.mv_path, os.listdir(self.mv_path)[0]))
        )

        try:
            pp.runpp(net)
        except Exception as e:
            raise Exception(f'creation of pandapower net fail and could not converge. {e}')
        else:
            self.write_net(net)
            plotly.simple_plotly(
                net=net,
                figsize=10,
                aspectratio=(1, 1),
                line_width=2,
                bus_size=5,
                ext_grid_size=20,
                auto_open=False,
                filename=os.path.join(self.work_path, 'net_plot.html')
            )
            return net

    def set_load(self, load_profile: pl.DataFrame, net: pp.pandapowerNet):
        net.load['p_mw'] = 0
        net.load['q_mvar'] = 0
        for load in load_profile.iter_rows(named=True):
            if load['bus'] in net.load['name'].to_list():
                net.load.loc[net.load['name'] == load['bus'], 'p_mw'] = float(load['p_mw'])
                net.load.loc[net.load['name'] == load['bus'], 'q_mvar'] = float(load['q_mvar'])
            else:
                logger.warning(f"Load at bus {load['bus']} does not exist in grid topology model.")
        # return net

    @decorator_timer
    def run(self):
        net = self.read_net()
        t0 = time.time()
        pp.runpp(net)
        dt = time.time() - t0
        logger.notice(f"[{datetime.now().isoformat()}] Profiler:  {round(dt,2)} s")
        t0 = time.time()
        pp.runpp(net)
        dt = time.time() - t0
        logger.notice(f"[{datetime.now().isoformat()}] Profiler:  {round(dt,2)} s")
        return
        for load_profile in self.load_profile_iter(from_date=datetime(year=2024, month=1, day=1),
                                                   to_date=datetime(year=2024, month=1, day=1)):
            self.set_load(load_profile, net)
            pp.runpp(net)
