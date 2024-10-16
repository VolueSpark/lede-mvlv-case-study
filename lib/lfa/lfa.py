from pandapower.plotting import plotly
from datetime import datetime
from typing import List, Any, Tuple
import pandapower as pp
import os, json, uuid
import polars as pl


FLEXIBILITY_MAX_USAGE_SCALE  = 0.5

from lib.lfa.topology import (
    Topology,
    UsagePoint,
    ConnectivityNode,
    PowerTransformer,
    AcLineSegment,
    Switch,
    GhostNodes
)

from lib.lfa import create_hash_from_hex
from lib.lfa.dataloader import DataLoader
from lib import logger, decorator_timer


PATH = os.path.dirname(os.path.abspath(__file__))


def pp_bus(bus: Any, net: pp.pandapowerNet, rated_kv: float=0):

    if type(bus) is str:
        if net.bus.loc[net.bus['mrid'] == bus].shape[0] == 1:
            return net.bus.loc[net.bus['mrid'] == bus].index.item()
        raise Exception(f"pp_bus raised exception for unresolved {len(net.bus.loc[net.bus['mrid'] == bus])} entries of bus={bus} detected net")

    if type(bus) is ConnectivityNode:
        if net.bus.loc[net.bus['mrid'] == bus.mrid].shape[0] == 1:
            return net.bus.loc[net.bus['mrid'] == bus.mrid].index.item()
        if net.bus.loc[net.bus['mrid'] == bus].shape[0] == 0:
            logger.warning(f'Bus mrid {bus.mrid} cannot be not resolved in pandapower net model. Add an artificial out-of-service bus.')
            return create_bus(
                bus=ConnectivityNode(
                    mrid=bus.mrid,
                    rated_kv=bus.rated_kv,
                    in_service=False
                ),
                net=net
            )
        raise Exception(f"pp_bus raised exception for unresolved bus creation entries for bus={bus.mrid} detected net")


def create_bus(bus: ConnectivityNode, net: pp.pandapowerNet):
    if ('mrid' not in net.bus.keys()) or (bus.mrid not in net.bus['mrid']):
        return pp.create_bus(
            net,
            name=f'bus_{bus.mrid}',
            mrid=bus.mrid,
            type='n',
            vn_kv=bus.rated_kv,
            in_service=bool(bus.rated_kv)
        )
    else:
        raise Exception(f'create_bus raised exception for multiple entries of bus={bus.mrid} in net')


def create_ext_grid(bus: ConnectivityNode, net: pp.pandapowerNet):
    create_bus(bus=bus, net=net)
    if ('mrid' not in net.ext_grid.keys()) or (bus.mrid not in net.ext_grid['mrid']):
        pp.create_ext_grid(
            net,
            name=f'slack_{bus.mrid}',
            mrid=bus.mrid,
            vm_pu=1.0,
            bus=pp_bus(bus=bus.mrid, net=net)
        )
    else:
        raise Exception(f'create_ext_grid raised exception for multiple external grid entries of bus={bus.mrid} in net')


def create_trafo(trafo: PowerTransformer, net: pp.pandapowerNet, in_service: bool=None):
    if ('mrid' not in net.trafo.keys()) or (trafo.mrid not in net.trafo['mrid']):
        if trafo.is_3w_trafo:
            pp.create_transformers3w_from_parameters(
                net,
                name=f'trafo_{trafo.name}',
                uuid=trafo.lv_bus.mrid, # keeping track of topology association
                mrid=trafo.mrid,
                hv_buses=[pp_bus(bus=trafo.hv_bus, net=net)],
                mv_buses=[pp_bus(bus=trafo.mv_bus, net=net)],
                lv_buses=[pp_bus(bus=trafo.lv_bus, net=net)],
                vn_hv_kv=trafo.hv_bus.rated_kv,
                vn_mv_kv=trafo.mv_bus.rated_kv,
                vn_lv_kv=trafo.lv_bus.rated_kv,
                sn_hv_mva=trafo.sn_mva, # TODO Verify number
                sn_mv_mva=trafo.sn_mva, # TODO Verify number
                sn_lv_mva=trafo.sn_mva, # TODO Verify number
                vk_hv_percent=10, # TODO Verify number
                vk_mv_percent=10, # TODO Verify number
                vk_lv_percent=10, # TODO Verify number
                vkr_hv_percent=5, # TODO Verify number
                vkr_mv_percent=5, # TODO Verify number
                vkr_lv_percent=5, # TODO Verify number
                pfe_kw=2, # TODO Verify number
                i0_percent=0.3, # TODO Verify number
                in_service=in_service if in_service is not None else trafo.in_service,
            )
        else:
            pp.create_transformer_from_parameters(
                net,
                name=f'trafo_{trafo.name}',
                uuid=trafo.lv_bus.mrid, # keeping track of topology association
                mrid=trafo.mrid,
                hv_bus=pp_bus(bus=trafo.hv_bus, net=net),
                lv_bus=pp_bus(bus=trafo.lv_bus, net=net),
                vn_hv_kv=trafo.hv_bus.rated_kv,
                vn_lv_kv=trafo.lv_bus.rated_kv,
                sn_mva=trafo.sn_mva,
                vkr_percent=5,  # TODO Verify number
                vk_percent=10,  # TODO Verify number
                pfe_kw=2,  # TODO Verify number
                i0_percent=0.4,  # TODO Verify number
                in_service=in_service if in_service is not None else trafo.in_service,
            )
    else:
        raise Exception(f'create_trafo raised exception for multiple trafo entries of trafo={trafo.name} in net')


def create_branch(branch: AcLineSegment, net: pp.pandapowerNet):
    if branch.has_impedance:
        if ('mrid' not in net.line.keys()) or (branch.mrid not in net.line['mrid']):
            pp.create_line_from_parameters(
                net,
                name=f'branch_{branch.name}',
                mrid=branch.mrid,
                from_bus=pp_bus(bus=branch.from_bus, net=net),
                to_bus=pp_bus(bus=branch.to_bus, net=net),
                length_km=1,  # TODO Verify number
                r_ohm_per_km=branch.r + 1e-5, # TODO Verify conditioning number added to prevent singularity
                x_ohm_per_km=branch.x +1e-5,
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
    if (type(element) in [Switch, AcLineSegment] and ('mrid' not in net.switch.keys() or element.mrid not in net.switch['mrid'])):
        pp.create_switch(
            net,
            name=f'switch_{element.name}',
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
        is_open=True,
        name=f'ghost_{uuid.uuid4().__str__()}'
    )
    create_switch(element=switch, net=net)


class LfaValidation:
    def __init__(self, topology: Topology):

        logger.debug(msg=f'Creating net for topology: {topology.uuid}')

        self.net = pp.create_empty_network(name=topology.uuid)

        for bus in topology.bus:
            create_bus(bus=bus, net=self.net)

        for bus in topology.slack:
            create_ext_grid(bus=bus, net=self.net)

        for trafo in topology.trafo:
            create_trafo(trafo=trafo, net=self.net, in_service=True)

        for branch in topology.branch:
            create_branch(branch=branch, net=self.net)

        for switch in topology.switch:
            create_switch(element=switch, net=self.net)

        for load in topology.load:
            create_load(load=load, net=self.net)

        for ghost in topology.ghost:
            create_ghost(ghost=ghost, net=self.net)

    @property
    def validate(self):
        try:
            pp.runpp(self.net)
            assert self.net['converged']
        except Exception as e:
            logger.exception(f'topology {self.net.name} failed so pass zero load-profile analysis')


class Lfa(DataLoader):
    def __init__(
            self,
            work_path: str,
            data_path: str,
            lv_path: str,
            mv_path: str
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
    def create_net(self, lv: List[Topology], mv: List[Topology]) -> pp.pandapowerNet:

        name = create_hash_from_hex(uuid_list=[mv_i.uuid for mv_i in mv])
        net = pp.create_empty_network(name=name)

        # add busses
        for i, mv_i in enumerate(mv):
            for j, mv_i_bus_j in enumerate(mv_i.bus):
                logger.debug(msg=f'[{i},{j}] mv topology.uuid={mv_i.uuid}, bus.mrid={mv_i_bus_j.mrid}')
                create_bus(bus=mv_i_bus_j, net=net)

        for i, lv_i in enumerate(lv):
            for j, lv_i_bus_j in enumerate(lv_i.bus):
                logger.debug(msg=f'[{i},{j}] lv topology.uuid={lv_i.uuid}, bus.mrid={lv_i_bus_j.mrid}')
                create_bus(bus=lv_i_bus_j, net=net)

        # add slack busses
        for i, mv_i in enumerate(mv):
            for j, mv_i_slack_j in enumerate(mv_i.slack):
                logger.debug(msg=f'[{i},{j}] mv topology.uuid={mv_i.uuid}, slack.mrid={mv_i_slack_j.mrid}')
                create_ext_grid(bus=mv_i_slack_j, net=net)

        # add trafo's
        for i, mv_i in enumerate(mv):
            for j, mv_i_trafo_j in enumerate(mv_i.trafo):
                logger.debug(msg=f'[{i},{j}] mv topology.uuid={mv_i.uuid}, trafo.mrid={mv_i_trafo_j.mrid} (in-service={mv_i_trafo_j.in_service})')
                if mv_i_trafo_j.in_service:
                    create_trafo(trafo=mv_i_trafo_j, net=net)

        # add branches
        for i, mv_i in enumerate(mv):
            for j, mv_i_branch_j in enumerate(mv_i.branch):
                logger.debug(msg=f'[{i},{j}] mv topology.uuid={mv_i.uuid}, branch.mrid={mv_i_branch_j.mrid}')
                create_branch(branch=mv_i_branch_j, net=net)

        for i, lv_i in enumerate(lv):
            for j, lv_i_branch_j in enumerate(lv_i.branch):
                logger.debug(msg=f'[{i},{j}] lv topology.uuid={lv_i.uuid}, branch.mrid {lv_i_branch_j.mrid}')
                create_branch(branch=lv_i_branch_j, net=net)

        # add switches
        for i, mv_i in enumerate(mv):
            for j, mv_i_switch_j in enumerate(mv_i.switch):
                logger.debug(msg=f'[{i},{j}] mv topology.uuid={mv_i.uuid}, switch.mrid={mv_i_switch_j.mrid}')
                create_switch(element=mv_i_switch_j, net=net)

        for i, lv_i in enumerate(lv):
            for j, lv_i_switch_j in enumerate(lv_i.switch):
                logger.debug(msg=f'[{i},{j}] lv topology.uuid={lv_i.uuid}, switch.mrid={lv_i_switch_j.mrid}')
                create_switch(element=lv_i_switch_j, net=net)

        # add loads
        for i, mv_i in enumerate(mv):
            for j, mv_i_load_j in enumerate(mv_i.load):
                logger.debug(msg=f'[{i},{j}] mv topology.uuid={mv_i.uuid}, load.mrid={mv_i_load_j.mrid}')
                create_load(load=mv_i_load_j, net=net)

        for i, lv_i in enumerate(lv):
            for j, lv_i_load_j in enumerate(lv_i.load):
                logger.debug(msg=f'[{i},{j}] lv topology.uuid={lv_i.uuid}, load.mrid={lv_i_load_j.mrid}')
                create_load(load=lv_i_load_j, net=net)

        # add ghost node
        for i, mv_i in enumerate(mv):
            for j, mv_i_ghost_j in enumerate(mv_i.ghost):
                logger.debug(msg=f'[{i},{j}] lv topology.uuid={mv_i.uuid}, ghost.mrid={mv_i_ghost_j.mrid}')
                create_ghost(ghost=mv_i_ghost_j, net=net)

        for i, lv_i in enumerate(lv):
            for j, lv_i_ghost_j in enumerate(lv_i.ghost):
                logger.debug(msg=f'[{i},{j}] lv topology.uuid={lv_i.uuid}, ghost.mrid={lv_i_ghost_j.mrid}')
                create_ghost(ghost=lv_i_ghost_j, net=net)

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
            mv=[read(topology_path=os.path.join(self.mv_path, mv_name)) for mv_name in os.listdir(self.mv_path)]
        )

        if not self.diagnose(net=net):
            raise Exception(f"creation of pandapower net failed and could not converge. See {os.path.join(self.work_path, 'diagnostics.json')}")
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

    def set_load(self, load_profile: pl.DataFrame, net: pp.pandapowerNet, activate_flex: bool = False):
        net.load['p_mw'] = 0.0
        net.load['q_mvar'] = 0.0
        for i, load in enumerate(load_profile.iter_rows(named=True)):
            if load['meter_id'] in net.load['name'].to_list():
                if self.flex_assets is not None and load['meter_id'] in self.flex_assets['meter_id']:

                    flex_meter = self.flex_assets.filter(pl.col('meter_id')==load['meter_id'])
                    max_usage_limit_mwh = flex_meter['max_usage_p_kw'].item()/1000
                    max_usage_limit_mvarh = flex_meter['max_usage_q_kvar'].item()/1000

                    logger.info(f"[{i+1}] Activate flexible asset {flex_meter['meter_id'].item()} in topology {flex_meter['uuid'].item()} for max usage: "
                                f"(P,Q)->({round(load['p_mw']*1000,2)},{round(load['q_mvar']*1000,2)})<=({round(max_usage_limit_mwh*1000,2)},{round(max_usage_limit_mvarh*1000,2)})  ")

                    net.load.loc[net.load['name'] == load['meter_id'], 'p_mw'] = min(load['p_mw'], max_usage_limit_mwh)
                    net.load.loc[net.load['name'] == load['meter_id'], 'q_mvar'] = min(load['q_mvar'], max_usage_limit_mvarh)
                else:
                    net.load.loc[net.load['name'] == load['meter_id'], 'p_mw'] = load['p_mw']
                    net.load.loc[net.load['name'] == load['meter_id'], 'q_mvar'] = load['q_mvar']

            else:
                logger.warning(f"[{i+1}] Load at bus {load['bus']} does not exist in grid topology model.")

    def parse_result(self, net: pp.pandapowerNet) -> dict:

        loss_percent = []
        for i, res_line in net.res_line.iterrows():

            ploss_kw = (net.line.loc[i]['r_ohm_per_km']*net.line.loc[i]['length_km']*res_line['i_ka']**2)
            vn_kv = max(net.bus.loc[net.line.loc[i]['from_bus']]['vn_kv'], net.bus.loc[net.line.loc[i]['to_bus']]['vn_kv'])
            p_kw = res_line['i_ka']*vn_kv

            loss_percent.append(round(ploss_kw/p_kw*100 if p_kw else 0,3))

        branch_data = {
            'branch_mrid':[mrid.replace('-','') for mrid in net.line['mrid'].to_list()],
            'loading_percent':net.res_line['loading_percent'].to_list(),
            'loss_percent': loss_percent
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
    def solve(self, net: pp.pandapowerNet):
        pp.runpp(net)

    def run_lfa(
            self,
            from_date: datetime,
            to_date: datetime = None,
            step_every: int = 1,
            flex_assets: pl.DataFrame=None
    ) -> Tuple[datetime, dict]:
        net = self.read_net()
        self.flex_assets = flex_assets

        for load_profile in self.load_profile_iter(
                from_date=from_date,
                to_date=to_date,
                step_every=step_every
        ):

            self.set_load(
                load_profile,
                net
            )
            self.solve(net=net)
            yield (load_profile['datetime'][0], self.parse_result(net=net))

    def diagnose(self, net: pp.pandapowerNet)->bool:
        diag = pp.diagnostic(net)
        diag_results = {
            'converged': net['converged'],
            'voltage_concerns': {},
            'impedance_concerns': {}
        }

        for element in diag['different_voltage_levels_connected'].keys():
            diag_results['voltage_concerns'][element] = []
            for index in diag['different_voltage_levels_connected'][element]:
                if element == 'lines':
                    mrid = net.line.loc[index]['mrid']
                    from_bus_index = net.line.loc[index]['from_bus']
                    to_bus_index = net.line.loc[index]['to_bus']
                elif element == 'switches':
                    mrid = net.switch.loc[index]['mrid']
                    from_bus_index = net.switch.loc[index]['bus']
                    to_bus_index = net.switch.loc[index]['element']

                diag_results['voltage_concerns'][element].append({'mrid': mrid,
                                                                  'from_bus': {net.bus.loc[from_bus_index]['mrid']: net.bus.loc[from_bus_index]['vn_kv']},
                                                                  'to_bus': {net.bus.loc[to_bus_index]['mrid']: net.bus.loc[to_bus_index]['vn_kv']}})

        for element in diag['impedance_values_close_to_zero']:
            for key, value in element.items():
                diag_results['impedance_concerns'][key] = []
                if key == 'line':
                    for index in value:
                        mrid = net.line.loc[index]['mrid']
                        r_ohm_per_km = net.line.loc[index]['r_ohm_per_km']
                        x_ohm_per_km = net.line.loc[index]['x_ohm_per_km']

                        diag_results['impedance_concerns'][key].append({'mrid': mrid,
                                                                        'r_ohm_per_km': f'{r_ohm_per_km:.3e}' ,
                                                                        'x_ohm_per_km': f'{x_ohm_per_km:.3e}'})

        with open(os.path.join(self.work_path, 'diagnostics.json'), 'w') as fp:
            json.dump(diag_results, fp)
        return diag_results['converged']
