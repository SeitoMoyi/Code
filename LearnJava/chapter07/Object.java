package chapter07;

public class Object {
    public static void main(String[] args){
        Person p = new Person();
        T t = new T();

        p.name = "jack";
        p.age = 10;
        t.altAge(p);
        // 引用传递的实质是地址赋值（复制了一份），因此altAge方法中的p是另一个指针，其无论怎么改变，也不会对main中的p的指向造成影响
        // altAge取得了p的地址，因此能够更改p中的元素
        System.out.println(p.age);//不会报错
    }
}
class Person{
    String name;
    int age;
}
class T{
    public void altAge(Person p){
        p.age = 100;
        p = null;
    }
}