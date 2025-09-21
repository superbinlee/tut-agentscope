# 使用Ubuntu 22.04作为基础镜像
FROM ubuntu:22.04

# 设置非交互模式，避免安装过程中出现交互提示
ENV DEBIAN_FRONTEND=noninteractive

# 更新包索引并安装必要的依赖
RUN apt-get update && apt-get install -y \
    curl \
    tar \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# 定义Node.js版本和下载地址
ENV NODE_VERSION=v22.19.0
ENV NODE_DOWNLOAD_URL=https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-x64.tar.xz

# 下载并安装Node.js
RUN curl -fsSL ${NODE_DOWNLOAD_URL} -o node.tar.xz \
    && mkdir -p /usr/local/node \
    && tar -xJf node.tar.xz -C /usr/local/node --strip-components=1 \
    && rm node.tar.xz

# 将Node.js添加到系统PATH
ENV PATH="/usr/local/node/bin:${PATH}"

# 验证Node.js和npm安装
RUN node -v && npm -v

# 全局安装agentscope/studio
RUN npm install -g @agentscope/studio

# 暴露端口3000
EXPOSE 3000

# 启动命令，运行as_studio
CMD ["as_studio"]
