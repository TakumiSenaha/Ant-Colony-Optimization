# 可変フェロモン最小値方式と可変揮発量方式の両方を用いたシミュレーション
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
from variable_min_pheromone import (
    set_pheromone_based_on_dimension,
    volitile_pheromone_based_on_dimension,
)
from variable_volatilization import volitile_pheromone_based_on_width


# 揮発時にwidthが小さいほど揮発量を大きくかつ
# 次元数によって可変なフェロモン最小値下回らないようにフェロモン揮発
def volitile_pheromone_based_on_dimension_and_width(
    self: Network, params: Params
) -> None:
    print("this is volitile_pheromone_based_on_dimension_and_width")
    for node in self.nodes:
        for link in node.neighbors.values():
            degree = len(node.neighbors)
            floor = (params.pheromone_min * 3) // (degree)
            # rate = params.volatility + (link.width / 500) # 世代200から上がっていきそうな兆しのあるparm
            rate = params.volatility * (0.1 ** ((100 - link.width) / 10))
            tmp = math.floor(link.pheromone * rate)
            if tmp < floor:
                link.pheromone = floor
            elif tmp > params.pheromone_max:
                link.pheromone = params.pheromone_max
            else:
                link.pheromone = tmp


def main(params: Params):

    # Networkクラスにset_pheromone_based_on_dimensionメソッド追加
    Network.set_pheromone_based_on_dimension = set_pheromone_based_on_dimension

    # Networkクラスのvolitile_pheromoneメソッドを差し替え
    Network.volitile_pheromone = volitile_pheromone_based_on_dimension_and_width

    # ネットワーク作成後にset_pheromone_based_on_dimension()を実行する手順を追加
    try:
        # DBLoggerインスタンス作成
        dblogger = DBLogger("user", "password", "localhost", "test_db", "5432")

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

        # ノードのフェロモンをノードのエッジ数によって変化させる
        simulation.network.set_pheromone_based_on_dimension(params)

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
            simulation.ant.hop_if_movable(params, generation_count)

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
        # Network modelを描画して保存したい
        dblogger.close()


if __name__ == "__main__":
    # パラメータを設定
    params = Params(
        num_nodes=20,
        optimal_route_length=6,
        volatility=0.99,
        pheromone_min=100,
        pheromone_max=2**20,
        ttl=100,
        bata=2,
        generation_limit=500,
        simulation_count=100,
    )

    with Pool() as p:
        p.map(main, [params] * params.simulation_count)


if __name__ == "__main__":
    # パラメータを設定
    params = Params(
        num_nodes=20,
        optimal_route_length=6,
        volatility=0.99,
        pheromone_min=100,
        pheromone_max=2**20,
        ttl=100,
        bata=2,
        generation_limit=500,
        simulation_count=100,
    )

    with Pool() as p:
        p.map(main, [params] * params.simulation_count)
