# build_squad_gap.py
接收SRC-GAP alignment作为输入，
构建squad用JSON训练/测试文件

输入格式：
```text
[CLS] s1 s2 ¶ s3 ¶ s4 [SEP] t1 [GAP] t2 [GAP] t3 [GAP] t4 [SEP]
```
即输入的MT中自带`[GAP]`token。
自然，label中直接将source单词对应到某个`[GAP]`。
当某个source单词在SRC-GAP中不存在对应，有三种策略进行数据补充。
分别是`skip, empty, src_mt`。目前来看只有`src_mt`能得到比较合理的结果。

# run_squad_align.py
因为是普通对应抽出训练过程，可以直接用`source-mt-align/run_squad_align.py`

# prediction2gap_info.py
在一般的prediction2align.py的基础上，可以直接根据指定的threshold输出pred.gap_tags。
（有超过threshold的对应点就是BAD，否则就是OK）
 