package chapter07;

public class PropertyDetails {
    public static void main(String[] args){
        P p = new P();
        p.say();
    }  
}
class P{
    public String name = "jack";// 属性可以加修饰符(public,protercted,private)，局部变量不可以
    public void say(){
        String name = "jacky";// 属性和局部变量可以重名，遵循就近原则
        System.out.println(name);
    }
}