# Question : i = i++

计算i = i++的值，不同语言结果不同。

```c++
#include<iostream>
using namespace std;
int main(){
	int i = 1;
	i = i++;
  //i = 1;
	//i++;
	cout << i << endl;
	return 0;
}
```

c++代码输出结果为2。

```c++
i = i++；
等价于
i = 1;
i++;
```

而在Java中运行结果为1。

```java
public static void main (String[] args){
	int i = 1;
	i = i++;
	System.out.println(i);
}
```

原因在于Java中会将赋值的内容用临时变量temp存储起来。

```java
i = i++;
等价于
temp = i;
i++;//先计算右侧的表达式
i = temp;//赋值的操作优先级最低，最后进行
```

