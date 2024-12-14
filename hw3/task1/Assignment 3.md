# Assignment 3

我的代码可以在github仓库中获取到[shuitatata/24Fall-NLPDL](https://github.com/shuitatata/24Fall-NLPDL)

## Task 1

### KV-cache 评测

我通过使用`generate`方法，并控制参数`use_cache`，来比较有无KV-cache情况下的性能。

使用gpt2模型，在4090显卡上进行实验。

进行5次实验后，推理时间和消耗显存大小如下：

| 是否使用KV-cache | 推理时间/s | 显存/GB |
| ---------------- | ---------- | ------- |
| 不使用KV-cache   | 18.4       | 11.8    |
| 使用KV-cache     | **7.6**    | **7.3** |

可以发现，使用KV-cache后，无论是推理时间还是显存占用都相较于naive的实现有了很大改善。这是因为KV-cache可以在构造key和value时，减少了两个大矩阵相乘的步骤，减少了时间和空间开销。

### 量化

这里的量化好像有两种理解，一种是使用**量化缓存**，见[使用缓存进行生成最佳实践 - Hugging Face 机器学习平台](https://hugging-face.cn/docs/transformers/kv_cache)，另一种则是对整个模型进行量化。

我首先尝试了第一种理解，但是在设置`cache_implementation`参数时，程序报告warning: This model does not support `Cache` instances。好像不支持使用量化缓存。

于是我按照第二种理解展开了实验。使用`quanto`后端，利用`quantize`接口对模型进行不同程度的量化，比较其推理速度和消耗显存。

使用gpt2模型，不使用KV-cache，在4090显卡上进行实验。

进行5次实验后，推理时间和消耗显存大小如下：

| 量化位数 | 推理时间/s | 显存/GB  |
| -------- | ---------- | -------- |
| int2     | 20.0       | **9.88** |
| int4     | 20.2       | 9.89     |
| int8     | **19.0**   | 9.89     |

可能是gpt2模型本身参数量就较小，模型大小不足1G，因此量化的作用不大。不管是推理时间还是显存占用都基本相同。

### 手工实现KV-cache

接下来我修改`customized_gpt2`的代码，主要思路是将之前的key和value存下来，与新的key和value拼接，构成完整的key和value，避免了每次都进行完整的矩阵乘法运算。

进行5次实验后，推理时间和消耗显存的对比如下：

| 是否使用KV-cache         | 推理时间/s | 显存/GB  |
| ------------------------ | ---------- | -------- |
| 不使用KV-cache           | 13.48      | 12.1     |
| 使用KV-cache（手工实现） | **11.48**  | **10.5** |

可以发现，手工实现的效果比不上官方实现的。但是相较于没有使用KV-cache的情况，推理时间和显存消耗都有了改善。


## Task 2

task2中，我评测了多种prompt技术对大模型数学能力的影响。

我使用deepseek的模型，gsm8k数据集，由于token限额原因，仅选取前500条数据参与评测。

结果如下：

|          | Naive    | Naive + few-shot | COT    | Reflexion | ICL+COT   |
| -------- | -------- | ---------------- | ------ | --------- | --------- |
| Accuracy | 30.4%    | 45.4%            | 84.2%  | 74.4%     | **93.2%** |
| 时间     | **20**'' | **20**''         | 6'49'' | 8'56''    | 3'43''    |

可以发现，ICL+COT的效果是最好的，其中COT发挥了比较大的作用。但是COT技术由于会产生更长的输出文本，因此推理时间的增长比较显著。

有意思的是，Reflexion比单纯使用COT的效果更差，这似乎比较违反直觉，因为Reflexion看起来是COT再加上“怀疑”，按理说效果应该不差于COT才对。

我们可以从以下样例尝试分析以下原因：

### 例子

#### 原始问题与答案：

> Josh decides to try flipping a house.  He buys a house for \$80,000 and then puts in \$50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?

> How much did the house cost?  The cost of the house and repairs came out to 80,000+50,000=\$<<80000+50000=130000>>130,000
> How much did the repairs increase the value of the house?  He increased the value of the house by 80,000\*1.5=<<80000\*1.5=120000>>120,000
> What is the new value of the house?  So the new value of the house is 120,000+80,000=\$<<120000+80000=200000>>200,000
> How much profit did he make?  So he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000

#### Naive和Naive + few-shot

二者都是直接给出答案，分别为220000和120000.

#### COT

仅使用COT时，我设计的prompt是这样的：

> Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?\n\nPlease provide a detailed step-by-step solution and conclude with the final numerical answer in the format '#### [answer]', only a single number.

这是模型给出的回复：

> To determine Josh's profit from flipping the house, we need to follow these steps:
>
> 1. \*\*Calculate the total cost of the house:\*\*
>    - Purchase price of the house: \$80,000
>    - Repairs cost: \$50,000
>    - Total cost = Purchase price + Repairs cost
>    - Total cost = \$80,000 + \$50,000 =\$130,000
>
> 2. \*\*Determine the increased value of the house:\*\*
>    - The value of the house increased by 150%.
>    - To find the new value, we first calculate 150% of the total cost.
>    - 150% of \$130,000 = 1.50 \* \$130,000 = \$195,000
>
> 3. \*\*Calculate the profit:\*\*
>    - Profit = New value of the house - Total cost
>    - Profit = \$195,000 - \$130,000 = \$65,000
>
> Therefore, Josh made a profit of $65,000.
>
> \#### 65000

可以看出，模型在尝试一步步地思考，但是在第2步的时候出现了错误，导致最终生成了错误的答案。

#### COT+ICL

这个设置下，在上文COT的基础上，我为模型还提供了四个例子以供参考。完整的prompt过于长，在这里不展示，具体结构就是在数据集中挑选了4个不参与评测的数据，放在了真实问题之前，供模型学习。

模型给出的回复如下：

> How much did Josh spend in total? \*\* Josh spent 80,000+50,000=\$<<80000+50000=130000>>130,000 in total.
> How much did the value of the house increase by? \*\* The value of the house increased by 150% of the original price, which is 1.5\*80,000=\$<<1.5*80000=120000>>120,000.
> What is the new value of the house? \*\* The new value of the house is 80,000+120,000=\$<<80000+120000=200000>>200,000.
> How much profit did Josh make?  \*\* Josh made a profit of 200,000-130,000=\$<<200000-130000=70000>>70,000.
> \#### 70000

我们可以明显发现，模型不仅给出了正确答案，而且还模仿了数据集中条目的结构，在形式上与我们给出的样本几乎一致。

#### Reflexion

Reflexion是在COT的基础上增加了一句“Afterward, reflect on your answer and explain why it makes sense. If you find any mistakes, correct them”，完整的prompt如下：

> Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?\n\nPlease provide a detailed step-by-step solution. **Afterward, reflect on your answer and explain why it makes sense. If you find any mistakes, correct them**, and provide the final answer in the format '#### [answer]'.

模型的回复为：

> To determine Josh's profit from flipping the house, we need to follow these steps:
>
> \1. \*\*Calculate the total cost of the house:\*\*
>    \- Purchase price of the house: \$80,000
>    \- Repairs: \$50,000
>    \- Total cost = Purchase price + Repairs = \$80,000 + \$50,000 = \$130,000
>
> \2. \*\*Determine the increased value of the house:\*\*
>    \- The value of the house increased by 150%. This means the new value is 100% (original value) + 150% (increase) = 250% of the original value.
>    \- The original value of the house after repairs is \$130,000.
>    \- New value = 250% of \$130,000 = (250/100) \* \$130,000 = 2.5 \* \$130,000 = \$325,000
>
> \3. \*\*Calculate the profit:\*\*
>    \- Profit = New value - Total cost = \$325,000 - \$130,000 = \$195,000
>
> \### Reflection:
> The calculation makes sense because Josh invested a total of \$130,000 into the house (purchase price + repairs), and the house's value increased by 150%, resulting in a new value of \$325,000. The difference between the new value and the total cost is the profit, which is \$195,000. This profit reflects the increase in the house's value due to the repairs and improvements made by Josh.
>
> \### Final Answer:
> \#### $195,000

在前面的COT阶段，模型不出意料地没有得到正确的结果，而在后面的Refelction中，模型也并没有找出自己的问题。我认为这是合理的，模型如果有辨别正确的能力，那么一开始也不会做错了。

但是我们可以尝试改进一下，让模型**始终假设**自己刚才的回答是错误的，并尝试找出错误之处。

现在的prompt如下：

> Josh decides to try flipping a house.  He buys a house for \$80,000 and then puts in \$50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?\n\nPlease provide a detailed step-by-step solution. **Afterward, we suspect your answer is incorrect, identify the mistake, correct it,** and provide the final answer in the format '#### [answer]', only a single number.

结果模型在第一遍给出了正确的结果，但是在反思阶段又改成了一个错误的答案。这几乎可以说明模型其实没有分辨对错的能力，越被质问越心虚。
