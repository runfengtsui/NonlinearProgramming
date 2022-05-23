本实验为华北电力大学(保定)2020-2021学年非线性规划课内实验，该实验实现了一维搜索的 0.618 法和斐波那契(Fibonacci)法，无约束非线性规划中的梯度下降(Gradient Descend)法、共轭梯度(Conjugate Gradient)法、拟牛顿(Newton)法、变尺度(DFP)法和步长加速法以及有约束非线性规划中的可行方向法。

在本实验中，有两个文件夹 `TwoVariablesFunction` 和 `nVariablesFunction`. 其中文件夹 `TwoVariablesFunction` 中是针对二元函数无约束和有约束非线性规划问题的程序。在该文件夹中的无约束非线性规划算法均使用符号求导的方法计算函数的梯度和 Hessian 矩阵，这两个函数在文件 `GradientMethod.py` 中。正是因为当自变量比较多时采用符号求导的方法计算梯度向量和 Hessian 矩阵比较复杂，所以这些程序比较适合二元函数的情形。有约束非线性规划算法 Zoutendijk 法是可行方向法中的一种，其通过构造线性规划问题求解出搜索方向，其他的可行方向法还有梯度投影法和既约梯度法，其中梯度投影法是利用投影矩阵构造搜索方向，既约梯度法是利用既约矩阵构造搜索方向。对于 Zoutendijk 方法中构造出的线性规划问题，采用 Python `scipy` 库中的 `linprog` 函数进行求解。另外，这里实现的 Zoutendijk 不仅限制只有两个自变量，而且约束条件必须只有一个，同时必须是线性约束条件。

文件夹 `nVariablesFunction` 中的程序是采用数值方法重新编写了梯度函数和 Hessian 阵的函数，这两个函数存储在文件 `NumericalGradient.py` 中。然后利用这两个函数代替原来的符号求导方法得到的梯度函数和 Hessian 阵函数，得到了针对一般的 $n$ 元函数的无约束非线性规划算法。另外，这里实现了步长加速法求解无约束非线性规划问题。

以上各种程序中用到一维搜索时，因为 Fibonacci 法需要构造一定长度的 Fibonacci 数列，当所需要的 Fibonacci 数列的项数较多时占用的空间较大，为节省内存，这里均采用 0.618 法实现一维搜索。

对于一维搜索问题，考虑两个单峰函数 $f(t)=e^{-t}+e^t$ 和 $f(t)=(\sin t)^6\tan (1-t)e^{30t}$, 分别使用 0.618 法和 Fibonacci 法求解，给出这两个问题的迭代过程如下图所示。

<center>
<img src="https://kristophertsui.gitee.io/figure/imgs/iteration_sequence_of_618.png" alt="0.618">
<img src="https://kristophertsui.gitee.io/figure/imgs/iteration_sequence_of_fibonacci.png" alt="fibonacci">
</center>


对于无约束非线性规划问题，选定测试函数 $f(x,y)=(1-x)^2+2(y-x^2)^2$, 分别使用梯度下降法、共轭梯度法、拟 Newton 法、变尺度法和步长加速法对该函数进行寻优，寻优的过程如下图所示(图片按从左到右、从上到下依次排列)。

<center>
<img src="https://kristophertsui.gitee.io/figure/imgs/iteration_sequence_of_gradient_descend.png" alt="GradientDescend">
<img src="https://kristophertsui.gitee.io/figure/imgs/iteration_sequence_of_conjugate_gradient.png" alt="ConjugateGradient">
<img src="https://kristophertsui.gitee.io/figure/imgs/iteration_sequence_of_quasi_newton.png" alt="Quasi-Newton">
<img src="https://kristophertsui.gitee.io/figure/imgs/iteration_sequence_of_DFP.png" alt="DFP">
<img src="https://kristophertsui.gitee.io/figure/imgs/iteration_sequence_of_step_acceleration.png" alt="StepAcceleration">
</center>

对于有约束非线性规划问题，目标函数为 $f(x,y)=x^2+y^2-4x-4y$, 约束条件为 $x+2y\le 4$, 寻优的过程如下图所示。

<center>
<img src="https://kristophertsui.gitee.io/figure/imgs/iteration_sequence_of_zoutendijk.png" alt="Zoutendijk">
</center>