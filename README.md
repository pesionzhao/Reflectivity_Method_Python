反射率法地震图正演python实现

参考书目： Seismic Wave Propagation in Stratified Media， 作者：Brian Kennett.

Fortran代码网址： http://www.quest-itn.org/library/software/reflectivity-method.html

原项目为Fortran代码，我通过参考Brian Kennett的书籍以及代码完成了python版本的重写，方便以后的反演调用

教材笔记网址: 

[chapter2耦合波](https://pesionzhao.github.io/SeismicInversion/Paper/CoupleWave/)

[chapter4震源](https://pesionzhao.github.io/SeismicInversion/Paper/Appendix/)

[chapter6递归求解反射系数](https://pesionzhao.github.io/SeismicInversion/Paper/ch6/)

[Chapter7层状半空间的响应](https://pesionzhao.github.io/SeismicInversion/Paper/Source/)

[chapter7求解地震记录](https://pesionzhao.github.io/SeismicInversion/Paper/Response/)

原项目描述：

作者：Brian Kennett

设计用于计算一组均匀固体层对点矩张量源激励的响应。使用慢频域合成的计算包括所有近场项。该模型包括衰减-常数Q（没有因果修正），在标准设置中多达500层。对每个距离进行慢度积分（目前允许2500个慢度点），然后使用FFT转换为时间。

本仓库的反射率法实现在不考虑上层分量信号的情况下与fast reflectivity method结果对比无误

代码还在整理中，后续也会给出与zoeppritz正演的结果对比， 如果对大家有用的话希望大家能给个star！

