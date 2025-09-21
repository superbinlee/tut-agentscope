# 安装agentscope/studio

```shell
# 基于ubuntu:22.04

# nodejs下载地址
https://nodejs.org/dist/v22.19.0/node-v22.19.0-linux-x64.tar.xz

# 安装agentscope/studio
npm install -g @agentscope/studio  # or npm install @agentscope/studio
# 启动命令 端口3000
as_studio
```

## 基于docker运行

```shell
# 构建镜像
docker build -t agentscope-studio .

# 运行容器，映射3000端口
docker run -p 3000:3000 agentscope-studio
```