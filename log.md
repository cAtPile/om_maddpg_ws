7-16
[18:48] done: 配置了git
        TODO:   1. 配置相关环境
                2. 构建ref
                3. 构建frame.md
                4. 跑通ros2的px4
        question: gym生成的轨迹规则是什么
        think: 生成被预测的轨迹，1.随机航点法。2.奥恩斯坦-乌伦贝克过程（Ornstein-Uhlenbeck Noise）等随机过程
                    后续再考虑更复杂的apf规则和神经网络
        plan: 今天跑通 ros2的px4

7-20
[14:45] talk: 有段时间没管了
        plan: 继续配环境，无gui的gazebo

[16:35] done: 改用gpu渲染，的确更流畅了

[16:44] done: 无头模式更流畅
        plan: 构建随机规则采集无人机数据的自动化流程<-按照claude到22.04装载deepseek
        think: 初期先做航点随机生成轨迹，无障碍物环境，构建轨迹批量生成和数据再现

[18:12] claud: 完成配置
        plan：用deepseek开始构建自动化流程
        
[18:58] plan: 先把px4相关内容配置上去

[20:10] doing: 正在编译
        plan：构建一个数据录制工具->构建一个数据再现工具->构建随机轨迹生成器

[20:35] recorder: 测试一下这个节点
        bug：gazebo掉了
        talk：今天能成功运行就下班，搞点实习的东西

[21:49] recorder: 测试成功
        TODO:   1. 再现工具
                2. 随机目标生成器
                3. 批量生成器
        question：轨迹生成器的规模如何确定
        talk:下班
        
