package chapter05;

import java.util.Scanner;

public class Switch01 {
    public static void main (String[] args){

        //switch语句需要break，否则会执行所有case下的语句，称为“穿透”
        //switch(表达式)，表达式的返回值必须是byte,int,char,enum,String,不能是double

        Scanner myScanner = new Scanner(System.in);
        char c1 = myScanner.next().charAt(0);
        myScanner.close();
        switch (c1) {
            case 'a':
                System.out.println("info1");
                break;
            case 'b':
                System.out.println("info2");
                break;
            case 'c':
                System.out.println("info3");
                break;
            default:
                System.out.println("Error");
                break;
        }
        System.out.println("Continue");
    }
}