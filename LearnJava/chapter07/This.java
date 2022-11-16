package chapter07;

public class This {
    public static void main(String[] args){
        Pe p1 = new Pe("king");
        Pe p2 = new Pe("king", 60);
        System.out.println("p1:" + "\t" + p1.name + "\t" + p1.age);
        System.out.println("p2:" + "\t" + p2.name + "\t" + p2.age);
        System.out.println("p1和p2比较的结果=" + p1.ComPe(p2));
    }
}

class Pe{
    String name;
    int age;

    // this关键字类似python中的self，可以使用this来应对构造器的局部变量与属性重名的情况。
    // 在没有this的情况下，默认使用的是局部变量的值（就近原则），而 this.属性 是特指该类中的属性。
    public Pe(String name, int age){
        this.name = name;
        this.age = age;
    }

    // this可以在构造器中访问其他构造器，语法为 this(参数列表)，且这个语句必须放在第一条！

    public Pe(String name){
        this(name, 60);
    }

    public boolean ComPe(Pe p){
        return this.name.equals(p.name) && this.age == p.age;
    }
}