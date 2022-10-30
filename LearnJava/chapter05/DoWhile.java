package chapter05;

public class DoWhile {
    public static void main(String[] args){
        int i = 1;
        //do while 不论如何，先做一次do（先执行，后判断）
        do{
            System.out.println("Hello!");
            i++;
        }while(i <=1);
    }
}
