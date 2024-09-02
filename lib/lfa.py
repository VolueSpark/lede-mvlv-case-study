import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple, Union
import pandapower as pp
import polars as pl
import os, json

from lib.schemas.topology import Topology
from lib.schemas.schema import UsagePoint
from lib import logger

PATH = os.path.dirname(os.path.abspath(__file__))


def read_topology_data(topology_path: str):
    with open(os.path.join(topology_path), 'r') as fp:
        data = json.load(fp)
        return Topology(**data)


class DataLoader():
    def __init__(self, ami_data_path: str, workspace_path: str, loads: List[UsagePoint]):
        df = pl.DataFrame()
        self.data_path = os.path.join(workspace_path, 'data')
        if not os.path.exists(self.data_path):
            for load in loads:
                if not os.path.exists(os.path.join(ami_data_path, load.meter_id)):
                    logger.warning(f"Cannot load ami data for meter {load.meter_id}")
                else:
                    df = df.vstack(pl.read_parquet(os.path.join(ami_data_path, load.meter_id)).with_columns(bus=pl.lit(load.bus)))
            (df.with_columns(((pl.col('import_kwh')-pl.col('export_kwh'))/1e3).alias('p_mw'),
                             ((pl.col('import_kvah')-pl.col('export_kvah'))/1e3).alias('q_mvar'))
             .with_columns(((pl.col('p_mw')**2+pl.col('q_mvar')**2)**0.5).alias('s_mva'),)
             .drop('import_kwh', 'export_kwh', 'import_kvah', 'export_kvah')
             .write_parquet(self.data_path))

    def load_profile_iter(self, from_date: datetime, to_date: datetime=None):
        if to_date is None:
            to_date = from_date

        df= pl.read_parquet(self.data_path).filter(pl.col('datetime').is_between(from_date, to_date))
        # pl.read_parquet(self.data_path).group_by('meter_id').agg(peak_s_mva=pl.col('s_mva').max()).sort(by='peak_s_mva', descending=True)
        for batch in df.partition_by('datetime'):
            yield batch


class Lfa(DataLoader):
    def __init__(self,
                 medium_voltage_path: str,
                 low_voltage_path: str,
                 ami_data_path: str,
                 workspace_path: str):
        self.mv_path = medium_voltage_path
        self.lv_path = low_voltage_path
        self.work_path = workspace_path
        self.data_path = ami_data_path
        self.net_path = os.path.join(self.work_path, 'net.sqlite')

        if not os.path.exists(self.work_path):
            os.makedirs(self.work_path, exist_ok=True)
        if not os.path.exists(self.net_path):
            (mv_topology, lv_topology) = self.validate_topology()
            self.create_mv_lv_net(
                mv_topology=mv_topology,
                lv_topology=lv_topology
            )


    def validate_topology(self) -> Tuple[Topology, List[Topology]]:
        lv_topology_list = []
        with open(os.path.join(self.mv_path, os.listdir(self.mv_path)[0]), 'r') as fp:
            mv_topology = Topology(**json.load(fp))

        lv_slack_tiein = []
        mv_slack_busses = [trafo.hv_bus for trafo in mv_topology.trafo]

        for index, lv_topology_file in enumerate(os.listdir(self.lv_path)):
            try:
                with open(os.path.join(self.lv_path, lv_topology_file), 'r') as fp:
                    lv_topology = Topology(**json.load(fp))
                    net = self.create_lv_net(topology=lv_topology)
                    pp.runpp(net)
                    mv_topology_trafo = [trafo for trafo in mv_topology.trafo if lv_topology.slack[0].bus == trafo.hv_bus]
                    assert len(mv_topology_trafo)==1, f'Cannot establish a mv-lv slack tie-in for neighbourhood {lv_topology.uuid}'

                    if lv_topology.slack[0].bus not in mv_slack_busses:
                        raise Exception(f'Neigborhood {lv_topology.uuid} has no region tie-in and will be discarded. Trafo is'
                                        f'configured out of service')

                    # put trafo in service as direct MV-LV tie-in can be established
                    mv_topology_trafo[0].in_service = lv_topology.trafo[0].in_service = True
                    lv_topology_list.append(lv_topology)

            except Exception as e:
                logger.exception(f"[{index}] Validation of {lv_topology_file} for single LV LFA failed [{e}]")

        return (mv_topology, lv_topology_list)


    def create_mv_lv_net(self, mv_topology: Topology, lv_topology: List[Topology]) -> pp.pandapowerNet:
        mv_topology = next(iter(mv_topology_iter))
        for lv_topology in lv_topology_iter:
            pass


    @staticmethod
    def create_net(topology: Topology) -> pp.pandapowerNet:

            net = pp.create_empty_network(name=topology.uuid)

            for bus in topology.bus:
                pp.create_bus(net, name=bus.bus, type='n', vn_kv=bus.rated_kv, in_service=bus.in_service)

            for bus in topology.slack:
                pp.create_bus(net, name=bus.bus, type='n', vn_kv=bus.rated_kv, in_service=bus.in_service)
                pp.create_ext_grid(net, name=bus.bus, vm_pu=1.0, bus=net.bus.loc[net.bus['name'] == bus.bus].index.item())

            for trafo in topology.trafo:

                pp.create_transformer_from_parameters(net,
                                                      name=trafo.mrid,
                                                      hv_bus=net.bus.loc[net.bus['name'] == trafo.hv_bus].index.item(),
                                                      lv_bus=net.bus.loc[net.bus['name'] == trafo.lv_bus].index.item(),
                                                      sn_mva=trafo.sn_mva,
                                                      vn_hv_kv=trafo.vn_hv_kv,
                                                      vn_lv_kv=trafo.vn_lv_kv,
                                                      in_service=trafo.in_service,
                                                      vkr_percent=5,  # TODO Verify number
                                                      vk_percent=10,  # TODO Verify number
                                                      pfe_kw=2,  # TODO Verify number
                                                      i0_percent=3  # TODO Verify number
                                                      )


            for branch in topology.branch:
                if branch.has_impedance:
                    pp.create_line_from_parameters(net,
                                                   name=branch.mrid,
                                                   from_bus=net.bus.loc[net.bus['name'] == branch.from_bus].index.item(),
                                                   to_bus=net.bus.loc[net.bus['name'] == branch.to_bus].index.item(),
                                                   length_km=1,  # TODO Verify number
                                                   r_ohm_per_km=branch.r,
                                                   x_ohm_per_km=branch.x,
                                                   c_nf_per_km=200,  # TODO Verify number
                                                   max_i_ka=1,  # TODO Verify number
                                                   in_service=True
                                                   )
                else:
                    pp.create_switch(net,
                                     name=branch.mrid,
                                     bus=net.bus.loc[net.bus['name'] == branch.from_bus].index.item(),
                                     element=net.bus.loc[net.bus['name'] == branch.to_bus].index.item(),
                                     et='b',
                                     closed=True
                                     )

            for switch in topology.switch:
                pp.create_switch(net,
                                 name=switch.mrid,
                                 bus=net.bus.loc[net.bus['name'] == switch.from_bus].index.item(),
                                 element=net.bus.loc[net.bus['name'] == switch.to_bus].index.item(),
                                 et='b',
                                 closed=not bool(switch.is_open)
                                 )

            for load in topology.load:
                pp.create_load(net,
                               name=load.bus,
                               bus=net.bus.loc[net.bus['name'] == load.bus].index.item(),
                               p_mw=0.0)

            for ghost in topology.ghost:
                pp.create_switch(net,
                                 name=ghost.mrid,
                                 bus=net.bus.loc[net.bus['name'] == ghost.from_bus].index.item(),
                                 element=net.bus.loc[net.bus['name'] == ghost.to_bus].index.item(),
                                 et='b',
                                 closed=False
                                 )

            return net

        #if not os.path.isfile(os.path.join(self.workspace_path,'data')):
        #    df = pl.DataFrame()
        #    for load in topology.load:
        #        pl.read_parquet(os.path.join(self.workspace_path,'../../data/ami/bronze/'))

    @staticmethod
    def run_lfa(net: pp.pandapowerNet) -> pp.pandapowerNet:
        pp.runpp(net)
        return net

    def write_net(self, net: pp.pandapowerNet):
        if os.path.exists(os.path.join(self.work_path,'net.sqlite')):
            os.remove(os.path.join(self.work_path,'net.sqlite'))
        pp.to_sqlite(net, os.path.join(self.work_path,'net.sqlite'))

    def read_net(self) -> pp.pandapowerNet:
        if os.path.isfile(os.path.join(self.work_path,'net.sqlite')):
            return pp.from_sqlite(os.path.join(self.work_path,'net.sqlite'))

    def set_load(self, load_profile: pl.DataFrame, net: pp.pandapowerNet):
        net.load['p_mw'] = 0
        net.load['q_mvar'] = 0
        for load in load_profile.iter_rows(named=True):
            if load['bus'] in net.load['name'].to_list():
                net.load.loc[net.load['name']==load['bus'], 'p_mw'] = float(load['p_mw'])
                net.load.loc[net.load['name']==load['bus'], 'q_mvar'] = float(load['q_mvar'])
            else:
                logger.warning(f"Load at bus {load['bus']} does not exist in grid topology model.")
        #return net

    def run(self):
        net = self.read_net()
        for load_profile in self.load_profile_iter(from_date=datetime(year=2024, month=1, day=1),
                                                   to_date=datetime(year=2024, month=1, day=1)):
            self.set_load(load_profile, net)
            pp.runpp(net)

    def plot(self, pp_plotly=True):
        if pp_plotly:
            from pandapower.plotting import plotly
            net = self.read_net()
            fig = plotly.simple_plotly(net=net, figsize=10, aspectratio=(1, 1),line_width=2, bus_size=5, ext_grid_size=20, auto_open=False)
            fig.update_layout(width=int(2000), height=int(1500))
            fig.show()
        else:
            from pandapower.plotting import simple_plot
            net = self.read_net()
            fig, ax = plt.subplots(1,1)
            simple_plot(net,
                        ax=ax,
                        trafo_size=0.2,
                        load_size=0.1,
                        switch_size=0.1,
                        line_width=0.1,
                        bus_size=0.1,
                        ext_grid_size=0.5,
                        plot_line_switches=True)
            fig.savefig(os.path.join(self.workspace_path,'plot.png'))








