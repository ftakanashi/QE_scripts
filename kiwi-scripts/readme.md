适用于老版本（0.1.3）的OpenKiwi。
文档：https://unbabel.github.io/OpenKiwi/quickstart.html
老版本repo: https://github.com/Unbabel/OpenKiwi/tree/4834561cc54d1fbd96b194f8fc33d75e29e5132e

流程是先用parallel corpus训练predictor。
再用QE数据训练predictor-estimator。
最后用QEtest数据进行predict。

通常只要将元项目中的experiments中的train_predictor.yaml, train_estimator.yaml, predict_estimator.yaml几个运行的配置文件
给拿出来，将数据什么的修改到自己的路径下，然后跑就行了。

注意几个坑
```text
第一，如果要进行双端QE（即source和MT两边tag都想预测），则需训练两个方向的两个predictor。

第二，train_predictor.yaml中的extend-source-vocab和extend-target-vocab是将QE中的词融入到NMT模型训练中，有助于提高QE表现。

第三，train_estimator.yaml中默认只进行MT word tag的预测。需要手动控制几个配置中的项来决定训练。
通常，正向训练时预测MT tags，因此打开predict-target和predict-gaps
反向训练时预测Source tags，因此打开predict-source。除了这个，反向训练的时候应该指明加载的是预测source的predictor，所以要将默认的
load-pred-target注释，换成load-pred-source指向训练完成的预测source的Predictor模型。

此外，这个配置文件默认使用WMT18以前的MT tag格式，即默认不含有gap tags。如果要预测gap tags，则需要将wmt18-format设置为true。
```