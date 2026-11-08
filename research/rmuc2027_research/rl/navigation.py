"""Exact static reachability preparation for repeated RL target queries.

Training-only: portable submissions use the same query on demand. Preparing
public static geometry does not inspect a referee frame or another robot.
"""
from collections import OrderedDict
from threading import RLock
from types import MappingProxyType
import time
from .portable import SdkNavigator, _sdk

_TABLES=OrderedDict()
_LOCK=RLock()
_MAX_TABLES=8


class PreparedNavigator(SdkNavigator):
    def __init__(self,grid_m=.25):
        super().__init__(grid_m)
        self.preparation_seconds=0.

    def resolve_goal(self,robot,proposed,arena):
        folded=robot.get('transform_progress',0)>=.99
        grid_key=(arena['layout_revision'],robot['team'],folded,robot.get('radius',.28),self.grid_m)
        with _LOCK:
            graph=_sdk._prepare_chapter4(arena,robot['team'],folded,robot.get('radius',.28)+.04)
            graph_key=(id(graph),graph[2])
            table_key=(graph_key,grid_key)
            table=_TABLES.get(table_key)
            if table is not None:_TABLES.move_to_end(table_key)
        self.precomputed_reachability={} if table is None else table
        # This first query also constructs the exact same grid/components as
        # the portable navigator. Its result is never replaced by preparation.
        goal=super().resolve_goal(robot,proposed,arena)
        if table is None:
            with _LOCK:
                table=_TABLES.get(table_key)
                if table is None:
                    started=time.perf_counter()
                    component,nodes=self.graph_components[graph_key][0]
                    entries={}
                    for strict in (True,False):
                        for point in self.grids[grid_key]:
                            entries[(graph_key,point,strict,component)]=any(
                                graph[2](node,point,(),strict) for node in nodes)
                    table=MappingProxyType(entries)
                    _TABLES[table_key]=table
                    while len(_TABLES)>_MAX_TABLES:_TABLES.popitem(last=False)
                    self.preparation_seconds+=time.perf_counter()-started
                else:_TABLES.move_to_end(table_key)
            self.precomputed_reachability=table
        return goal
