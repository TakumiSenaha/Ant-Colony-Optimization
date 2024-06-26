# 可変揮発量方式
# 帯域の大きさによって揮発量を変化させる
# TODO 揮発時ににwidthが小さいほど揮発量を大きくするよう変更
import math
import random
import traceback
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Self, Tuple, cast

import psycopg2

from aco_base_small_model import (
    Ant,
    DBLogger,
    Interest,
    Link,
    Network,
    Node,
    Packet,
    Params,
    Rand,
    Simulation,
)

# 揮発時にwidthが小さいほど揮発量を大きくする


def volitile_pheromone_based_on_width(self: Network, params: Params) -> None:
    print("this is volitile_pheromone_based_on_width")
    for node in self.nodes:
        for link in node.neighbors.values():
            rate = 0.89 + (link.width / 1000)
            tmp = math.floor(link.pheromone * rate)
            if tmp < params.pheromone_min:
                link.pheromone = params.pheromone_min
            elif tmp > params.pheromone_max:
                link.pheromone = params.pheromone_max
            else:
                link.pheromone = tmp


def main(params: Params):

    # Networkクラスのvolitile_pheromoneメソッドを差し替え
    Network.volitile_pheromone = volitile_pheromone_based_on_width

    # 接続先DB以外はもとのmain関数と同じ
    try:
        # DBLoggerインスタンス作成
        dblogger = DBLogger(
            "asaken_n40", "asaken_N40", "localhost", "simulation", "5432"
        )

        dblogger.connect()

        # パラメータを登録&パラメータIDを取得
        params.id = dblogger.insert_conflict(
            params.generate_insert_or_return_id_query()
        )
        print(params.id)
        if params.id is None:
            params.id: int = dblogger.fetch_result(params.generate_select_query())[0][0]
            print(params.id)

        # Simulationインスタンス作成
        simulation = Simulation(dblogger, params)

        # シミュレーションを登録&シミュレーションIDを取得
        simulation.id = dblogger.insert_and_get_id(simulation.generate_insert_query())

        # 任意の個数ノードインスタンスを作成
        simulation.network.yield_nodes(params)

        # BAモデルになるようにノードを接続
        simulation.network.make_ba_model(params, 3)

        # 最適ルートを作成
        simulation.network.make_optimal_route(params)

        # ノードを登録&ノードIDを取得
        for node in simulation.network.nodes:
            node.id = dblogger.insert_and_get_id(
                node.generate_insert_query(simulation.id)
            )

        # 任意の回数Generationを繰り返す
        for generation_count in range(params.generation_limit):

            # Genaerationを登録&GenerationIDを取得
            generation_id = dblogger.insert_and_get_id(
                f"INSERT INTO generations (simulationid, generation_count) VALUES ({simulation.id},{generation_count});"
            )

            # Connectionsを登録
            for startnode in simulation.network.nodes:
                for endnode, link in node.neighbors.items():
                    dblogger.insert_and_get_id(
                        f"INSERT INTO connections (GenerationID, StartNodeID, EndNodeID, Pheromone, Width) VALUES ({generation_id},{startnode.id},{endnode.id},{link.pheromone},{link.width});"
                    )

            # antとinterestの生成
            simulation.ant = Ant(
                simulation.network.start_node, simulation.network.end_node
            )
            simulation.interest = Interest(
                simulation.network.start_node, simulation.network.end_node
            )

            # antの移動
            simulation.ant.hop_if_movable(params)

            # 目的地に到達していたらフェロモン付加
            if simulation.ant.is_at_destination():
                simulation.network.add_pheromone_to_ant_route(simulation.ant)

            # antの結果を登録
            dblogger.execute_query(simulation.ant.get_insert_query(generation_id))

            # antをNoneにして消去
            simulation.ant = None

            # interestの移動
            simulation.interest.hop_if_movable(params)

            # interestの結果を登録
            dblogger.execute_query(simulation.interest.get_insert_query(generation_id))

            # interestをNoneにして消去
            simulation.interest = None

            # フェロモン揮発
            simulation.network.volitile_pheromone(params)

        dblogger.commit()

    except Exception as e:
        print(e)
        dblogger.rollback()
        print(traceback.format_exc())

    finally:
        dblogger.close()


if __name__ == "__main__":
    # パラメータを設定
    params = Params(
        num_nodes=5,
        optimal_route_length=2,
        volatility=0.99,
        pheromone_min=100,
        pheromone_max=2**20,
        ttl=100,
        bata=1,
        generation_limit=2,
        simulation_count=1,
    )

    with Pool() as p:
        p.map(main, [params] * params.simulation_count)
    params = Params(
        num_nodes=5,
        optimal_route_length=2,
        volatility=0.99,
        pheromone_min=100,
        pheromone_max=2**20,
        ttl=100,
        bata=1,
        generation_limit=2,
        simulation_count=1,
    )

    with Pool() as p:
        p.map(main, [params] * params.simulation_count)
