package chapter07;

public class ConstructorDetail {
    public static void main(String[] args){
        Pers p1 = new Pers("king",40);
        Pers p2 = new Pers("tom");
    }
}

class Pers{
    // 1.一个类可以定义多个不同的构造器，即构造器的重载
    // 2.构造器名和类名要相同。
    // 3.构造器没有返回值。
    // 4.构造器是完成对象的初始化，并不是创建对象。
    // 5.在创建对象时，系统自动调用该类的构造器。
    String name;
    int age;
    public Pers(String pName, int pAge){
        name = pName;
        age = pAge;
    }
    public Pers(String pName){
        name = pName;
    }
}

class Dog{
    // 6.如果程序员没有定义构造器，系统会自动生成一个默认无参构造器（默认构造器），比如 Dog(){}，使用javap指令反编译查看。
    // 7.一旦定义了构造器，默认构造器就被覆盖，无法使用默认的无参构造器，除非显式定义一下，如 Dog(){}。(很重要!)
    /*
        默认构造器
        Dog(){

        }
     */
    String name;
    public Dog(){}
    public Dog(String dName){
        name = dName;
    }
}