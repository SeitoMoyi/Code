package chapter07;

public class Constructor {
    public static void main(String[] args){
        Per p1 = new Per("smith", 80);
        System.out.println("p1的信息如下");
        System.out.println("p1.name = " + p1.name);
        System.out.println("p1.age = " + p1.age);
    }
}
class Per{
    String name;
    int age;
    // 构造器
    // 1.构造器没有返回值，也不能写void。
    // 2.构造器的名称和类名Per必须一样。
    // 3.(String pName, int pAge)是构造器的形参列表，规则和成员方法一样。
    public Per(String pName, int pAge){
        System.out.println("====构造器被调用，完成对象属性的初始化====");
        name = pName;
        age = pAge;
    }
}
