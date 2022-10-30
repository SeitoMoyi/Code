package chapter05;

import java.util.Scanner;

public class If01 {
    public static void main(String[] args){
        Scanner myScanner = new Scanner(System.in);
        int age = myScanner.nextInt();
        myScanner.close();
        if(age >= 18){
            System.out.println("You're an adult.");
        }else{
            System.out.println("Enjoy!");
        }
    }
}
