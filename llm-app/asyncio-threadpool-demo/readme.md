# asyncio+fastapi异步多线程测试

asyncio简介

- asyncio 是用来编写 **并发** 代码的库，使用 **async/await** 语法

- asyncio 被用作多个提供高性能 Python 异步框架的基础，包括网络和网站服务，数据库连接库，分布式任务队列等等。

- asyncio 往往是构建 IO 密集型和高层级 **结构化** 网络代码的最佳选择。



uvicorn简介

- uvicorn是一个基于asyncio开发的一个轻量级高效的web服务器框架

- uvicorn 设计的初衷是想要实现两个目标：
  - 使用uvloop和httptools 实现一个极速的asyncio服务器
  - 实现一个基于ASGI（异步服务器网关接口）的最小应用程序接口。

fastapi简介

- FastAPI 是一个用于构建 API 的现代、快速（高性能）的 web 框架，专为在 Python 中构建 RESTful API 而设计

**三者的关系**

- uvicorn部署fastapi
- uvicorn将请求主线程交给fastapi,
- fastapi再把丢请求给asyncio的协程处理
- 从而，主线程可以一直可以接受请求，协程执行完了，再交给主线程