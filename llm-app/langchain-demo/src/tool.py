from fastapi import FastAPI, HTTPException, Depends
import mysql.connector
from mysql.connector import Error
import mysql.connector
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field
from fastapi import Depends, HTTPException
from langchain  import SQLDatabase

app = FastAPI()


# 依赖项函数，用于创建数据库连接
def get_mysql_connection(host: str, user: str, password: str, db: str):
    try:
        # 尝试连接到 MySQL 数据库
        mysql_uri = f'mysql+pymysql://{user}:{password}@{host}:3306/{db}'
        db = SQLDatabase.from_uri(mysql_uri)
        return db
    except Error as e:
        raise HTTPException(status_code=400, detail=str(e))


class TableInfo(BaseModel):
    table_name: str = Field(..., title="Table Name", description="Name of the table.")
    columns: List[Tuple[str, str]] = Field(..., title="Columns", description="List of columns with their types.")
    sample_data: Optional[List[Tuple]] = Field(None, title="Sample Data", description="First three rows of table data.")


@app.get("/get-schema/")
def get_schema(db: str, connection: mysql.connector.MySQLConnection = Depends(get_mysql_connection)):
    return {"schema": connection.get_table_info()}


@app.post("/execute-query/")
def execute_query(query: str, connection: mysql.connector.MySQLConnection = Depends(get_mysql_connection)):
    try:
        result = connection.run(query)
    except:
        result = '查询失败，请检查输入的sql语句是否正确'
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

