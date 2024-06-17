import psycopg2


def reset_database():
    try:
        # データベースに接続
        conn = psycopg2.connect(
            dbname="test_db", user="user", password="password", host="localhost"
        )
        cursor = conn.cursor()

        # テーブルの内容を削除
        cursor.execute(
            "TRUNCATE TABLE parameters, simulations, generations, nodes, connections, ants, interests RESTART IDENTITY CASCADE;"
        )

        # 変更をコミット
        conn.commit()

    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


# 関数を呼び出してデータベースをリセット
reset_database()
reset_database()
