package chapter04;
import java.util.Scanner;
public class Input {
    public static void main(String[] args) {
        Scanner myScanner =  new Scanner(System.in);
        System.out.println("请输入");
        String name = myScanner.next();
        int age = myScanner.nextInt();
        double sal = myScanner.nextDouble();
        System.out.println(name + " " + age + " " + sal);
        myScanner.close();
    }
}
