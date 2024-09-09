from pandapower.plotting import plotly
from datetime import datetime
from typing import List, Any, Tuple
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


def pp_bus(bus: str, net: pp.pandapowerNet):
    try:
        return net.bus.loc[net.bus['name'] == bus].index.item()
    except Exception as e:
        raise Exception(f'pp_bus raised exception for multiple entries of bus={bus} detected net')


def create_bus(bus: ConnectivityNode, net: pp.pandapowerNet):
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


def create_ext_grid(bus: ConnectivityNode, net: pp.pandapowerNet):
    create_bus(bus=bus, net=net)
    if bus.bus not in net.ext_grid['name']:
        pp.create_ext_grid(
            net,
            name=bus.bus,
            vm_pu=1.0,
            bus=pp_bus(bus=bus.bus, net=net)
        )
    else:
        raise Exception(f'create_ext_grid raised exception for multiple external grid entries of bus={bus.bus} in net')


def create_trafo(trafo: PowerTransformer, net: pp.pandapowerNet):
    if trafo.name not in net.trafo['name']:
        pp.create_transformer_from_parameters(
            net,
            name=trafo.name,
            mrid=trafo.mrid,
            hv_bus=pp_bus(bus=trafo.hv_bus, net=net),
            lv_bus=pp_bus(bus=trafo.lv_bus, net=net),
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


def create_branch(branch: AcLineSegment, net: pp.pandapowerNet):
    if branch.has_impedance:
        if branch.name not in net.line['name']:
            pp.create_line_from_parameters(
                net,
                name=branch.name,
                mrid=branch.mrid,
                from_bus=pp_bus(bus=branch.from_bus, net=net),
                to_bus=pp_bus(bus=branch.to_bus, net=net),
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
            element=branch, net=net
        )


def create_switch(element: Any, net: pp.pandapowerNet):
    if type(element) in [Switch, AcLineSegment] and element.mrid not in net.switch['name']:
        pp.create_switch(
            net,
            name=element.name,
            mrid=element.mrid,
            bus=pp_bus(bus=element.from_bus, net=net),
            element=pp_bus(bus=element.to_bus, net=net),
            et='b',
            closed=True if type(element) == AcLineSegment else not bool(element.is_open)
        )
    else:
        raise Exception(f'create_switch raised exception for multiple switch entries for switch={element.name} in net')


def create_load(load: UsagePoint, net: pp.pandapowerNet):
    if load.bus not in net.load['name']:
        pp.create_load(
            net,
            name=load.meter_id,
            mrid=load.mrid,
            bus=pp_bus(bus=load.bus, net=net),
            cfl_mrid=load.cfl_mrid,
            p_mw=0.0
        )
    else:
        raise Exception(f'create_load raised exception for multiple load entries for load={load.bus} in net')


def create_ghost(ghost: GhostNodes, net: pp.pandapowerNet):
    switch = Switch(
        mrid=ghost.mrid,
        from_bus=ghost.from_bus,
        to_bus=ghost.to_bus,
        is_open=False,
        name=f'branch_{uuid.uuid4().__str__()}'
    )
    create_switch(element=switch, net=net)


class LfaValidation:
    def __init__(self, topology: Topology):

        logger.debug(msg=f'Creating net for topology: {topology.uuid}')

        net = pp.create_empty_network(name=topology.uuid)

        for bus in topology.bus:
            create_bus(bus=bus, net=net)

        for bus in topology.slack:
            create_ext_grid(bus=bus, net=net)

        for trafo in topology.trafo:
            create_trafo(trafo=trafo, net=net)

        for branch in topology.branch:
            create_branch(branch=branch, net=net)

        for switch in topology.switch:
            create_switch(element=switch, net=net)

        for load in topology.load:
            create_load(load=load, net=net)

        for ghost in topology.ghost:
            create_ghost(ghost=ghost, net=net)

        try:
            pp.runpp(net)
        except Exception as e:
            logger.exception(f'topology {topology.uuid} failed so pass zero load-profile analysis')


class DataLoader():
    def __init__(
            self,
            lv_path: str,
            data_path: str,
            work_path: str
    ):

        self.data_path = os.path.join(work_path, 'data')
        os.makedirs(self.data_path, exist_ok=True)

        topology_list = os.listdir(lv_path)
        metadata = pl.DataFrame()
        for j, topology_j in enumerate(topology_list):
            if not os.path.exists(os.path.join(work_path, 'data', f"{topology_j}.parquet")):
                with open(os.path.join(lv_path, topology_j), 'r') as fp:

                    meter_list = [load['meter_id'] for load in json.load(fp)['load']]

                    df = pl.DataFrame()
                    for i, meter_i in enumerate(meter_list):
                        if os.path.exists(os.path.join(data_path, meter_i)):
                            df = df.vstack(pl.read_parquet(os.path.join(data_path, meter_i)))
                        else:
                            logger.warning(f"[meter {i+1} of {len(meter_list)}] {topology_j} has no data for meter {meter_i}")

                    if df.shape[0]:
                        df=(
                            df.with_columns(((pl.col('p_kwh_out') - pl.col('p_kwh_in')) / 1e3).alias('p_mw'),
                                            ((pl.col('q_kvarh_out') - pl.col('q_kvarh_in')) / 1e3).alias('q_mvar'))
                            .with_columns(((pl.col('p_mw') ** 2 + pl.col('q_mvar') ** 2) ** 0.5).alias('s_mva'))
                            .drop('p_kwh_in', 'p_kwh_out', 'q_kvarh_in', 'q_kvarh_out')
                        )

                        metadata = metadata.vstack(
                            (
                                df.group_by('meter_id')
                                .agg(
                                    (pl.col('p_mw').min() * 1000).round(1).alias('p_kw_min'),
                                    (pl.col('p_mw').max() * 1000).round(1).alias('p_kw_max'),
                                    (pl.col('p_mw').mean() * 1000).round(1).alias('p_kw_mean'),
                                    (pl.col('q_mvar').min() * 1000).round(1).alias('q_kvar_min'),
                                    (pl.col('q_mvar').max() * 1000).round(1).alias('q_kvar_max'),
                                    (pl.col('q_mvar').mean() * 1000).round(1).alias('q_kvar_mean'),
                                    pl.lit(topology_j).alias('uuid')
                                ).select('uuid', 'meter_id', 'p_kw_min', 'p_kw_mean', 'p_kw_max', 'q_kvar_min', 'q_kvar_mean', 'q_kvar_max')
                            )
                        )

                        df.write_parquet(os.path.join(self.data_path, f"{topology_j}.parquet"))

                        logger.info(f"[topology {j+1} of {len(topology_list)}] {topology_j} has been processed with {df.n_unique('meter_id')} unique meters")
                    else:
                        logger.exception(f"[topology {j+1} of {len(topology_list)}] {topology_j} has no available data")

        if not metadata.is_empty():
            metadata.write_parquet(os.path.join(work_path, f"metadata.parquet"))

    def load_profile_iter(self, from_date: datetime, to_date: datetime = None):
        if to_date is None:
            to_date = from_date

        data_list = os.listdir(self.data_path)
        data = pl.DataFrame()
        for data_file in data_list:
            data = data.vstack(
                pl.read_parquet(os.path.join(self.data_path, data_file))
                .filter(
                    pl.col('datetime').is_between(from_date, to_date)
                )
            )

        # pl.read_parquet(self.data_path).group_by('meter_id').agg(peak_s_mva=pl.col('s_mva').max()).sort(by='peak_s_mva', descending=True)
        for batch in data.partition_by('datetime'):
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
        super().__init__(
            lv_path=lv_path,
            data_path=data_path,
            work_path=work_path,
        )

        self.work_path = work_path
        self.mv_path = mv_path
        self.lv_path = lv_path
        self.net = self.read_net()


    @decorator_timer
    def create_net(self, lv: List[Topology], mv: Topology) -> pp.pandapowerNet:

        logger.debug(msg=f'Creating subnet for mv topology: {mv.uuid}')

        net = pp.create_empty_network(name=mv.uuid)

        # busses
        logger.debug(msg=f'compile mv busses for uuid {mv.uuid}')
        for mv_bus in mv.bus:
            create_bus(bus=mv_bus, net=net)

        for i, lv_i in enumerate(lv):
            logger.debug(msg=f'compile lv bus group {i + 1} for uuid {lv_i.uuid}')
            for lv_bus in lv_i.bus:
                create_bus(bus=lv_bus, net=net)

        # slack busses
        for i, mv_slack_bus in enumerate(mv.slack):
            logger.debug(msg=f'compile mv slack bus {i + 1} for bus {mv_slack_bus}')
            create_ext_grid(bus=mv_slack_bus, net=net)

        # trafo's
        for i, mv_trafo in enumerate(mv.trafo):
            logger.debug(msg=f'compile mv trafo {i + 1} with name: {mv_trafo.name}={mv_trafo.mrid} (in-service={mv_trafo.in_service})')
            if mv_trafo.in_service:
                create_trafo(trafo=mv_trafo, net=net)

        # branches
        for i, mv_branch in enumerate(mv.branch):
            logger.debug(msg=f'compile mv branch {i + 1} with name: {mv_branch.name}={mv_branch.mrid}')
            create_branch(branch=mv_branch, net=net)

        for i, lv_i in enumerate(lv):
            logger.debug(msg=f'compile lv branch group {i + 1} for uuid {lv_i.uuid}')
            for lv_branch in lv_i.branch:
                create_branch(branch=lv_branch, net=net)

        # switches
        for i, mv_switch in enumerate(mv.switch):
            logger.debug(msg=f'compile mv switch {i + 1} with name: {mv_switch.name}={mv_switch.mrid}')
            create_switch(element=mv_switch, net=net)

        for i, lv_i in enumerate(lv):
            logger.debug(msg=f'compile lv with group {i + 1} for uuid {lv_i.uuid}')
            for lv_switch in lv_i.switch:
                create_switch(element=lv_switch, net=net)

        # loads
        for i, mv_laod in mv.load:
            logger.debug(msg=f'compile mv load {i + 1} with name: {mv_laod.name}={mv_laod.mrid}')
            create_load(load=mv_laod, net=net)

        for i, lv_i in enumerate(lv):
            logger.debug(msg=f'compile lv load group {i + 1} for uuid {lv_i.uuid}')
            for lv_load in lv_i.load:
                create_load(load=lv_load, net=net)

        # ghost nodes
        for i, mv_ghost in mv.ghost:
            logger.debug(msg=f'compile mv ghost node {i + 1} with mrid: {mv_ghost.mrid}')
            create_ghost(ghost=mv_ghost, net=net)

        return net

    def write_net(self, net: pp.pandapowerNet):
        if os.path.exists(os.path.join(self.work_path, 'net.sqlite')):
            os.remove(os.path.join(self.work_path, 'net.sqlite'))
        pp.to_sqlite(net, os.path.join(self.work_path, 'net.sqlite'))

    def read_net(self) -> pp.pandapowerNet:
        if os.path.isfile(os.path.join(self.work_path, 'net.sqlite')):
            return pp.from_sqlite(os.path.join(self.work_path, 'net.sqlite'))

        @staticmethod
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
                figsize=4,
                aspectratio=(1920/1920, 1080/1920),
                line_width=1,
                bus_size=2,
                ext_grid_size=20,
                auto_open=False,
                filename=os.path.join(self.work_path, 'net_plot.html')
            )
            return net

    def set_load(self, load_profile: pl.DataFrame, net: pp.pandapowerNet):
        net.load['p_mw'] = 0.0
        net.load['q_mvar'] = 0.0
        for i, load in enumerate(load_profile.iter_rows(named=True)):
            if load['meter_id'] in net.load['name'].to_list():
                net.load.loc[net.load['name'] == load['meter_id'], 'p_mw'] = load['p_mw']
                net.load.loc[net.load['name'] == load['meter_id'], 'q_mvar'] = load['q_mvar']
            else:
                logger.warning(f"[{i+1}] Load at bus {load['bus']} does not exist in grid topology model.")

    def parse_result(self, net: pp.pandapowerNet) -> dict:

        branch_data = {
            'branch_mrid':[mrid.replace('-','') for mrid in net.line['mrid'].to_list()],
            'loading_percent':net.res_line['loading_percent'].to_list()
        }
        conform_load_data ={
            'cfl_mrid':[cfl_mrid.replace('-','') for cfl_mrid in net.load['cfl_mrid'].to_list()],
            'v_pu': [net.res_bus.iloc[pp_bus_index]['vm_pu'] for pp_bus_index in net.load['bus']]
        }
        trafo_data = {
            'trafo_mrid':[mrid.replace('-','') for mrid in net.trafo['mrid'].to_list()],
            'loading_percent':net.res_trafo['loading_percent'].to_list()
        }

        return {
            'branch': pl.from_dicts(branch_data),
            'conform_load': pl.from_dicts(conform_load_data),
            'trafo': pl.from_dicts(trafo_data)
        }


    @decorator_timer
    def run_lfa(
            self,
            from_date: datetime,
            to_date: datetime
    )->Tuple[datetime, dict]:
        net = self.read_net()
        for load_profile in self.load_profile_iter(from_date=from_date, to_date=to_date):
            self.set_load(load_profile, net)
            pp.runpp(net)
            yield (load_profile['datetime'][0], self.parse_result(net=net))
