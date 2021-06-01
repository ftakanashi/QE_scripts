# build_squad_gap_implicit.py
接收SRC-GAP alignment作为输入，
构建squad用JSON训练/测试文件

输入格式：
```text
[CLS] s1 s2 ¶ s3 ¶ s4 [SEP] t1 t2 t3 t4 [SEP]
```
输入中不带有`[GAP]`之类的特殊token，而是将某个gap对应到其后面的词
（最后一个gap的处理还没有完美实现。目前是暂时对应在最后一个单词但是答案text是空字符串）

同样，在SRC-GAP对应数据中，如果某个source单词没有找到任何对应的gap，那么针对其的处理方式有三种
skip表示直接不将这个单词的任何信息写入JSON
empty表示将这个单词的对应看做无答案的空示例写入JSON
align_to_word表示额外接收src_mt_align作为输入，并将这个单词在和mt word的对应写入JSON

# run_squad_align.py
目前这条思路还是简单的对应抽出，所以直接用`source-mt-align/run_squad_align.py`即可。

# prediction2gap_info_implicity.py
align_to_word模式下，如何区分输出结果中的to-gap对应和to-word对应还没想好。
其他两种模式，就是简单的阈值判断，然后输出