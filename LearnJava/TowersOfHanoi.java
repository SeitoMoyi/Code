import java.util.Scanner;

public class TowersOfHanoi {
    public static void main(String[] args) {
        System.out.println("Enter number of disks");
        Scanner cin = new Scanner(System.in);
        int n = cin.nextInt();
        cin.close();
        System.out.println("The move are:");
        moveDISKs(n, 'A','B','C');
    }
    public static void moveDISKs(int n, char fromTower, char toTower, char auxTower) {
        if (n == 1)
            System.out.println( "move disk " + n + " from " + fromTower + " to " + toTower);
        else {
            moveDISKs(n-1, fromTower, auxTower, toTower);
            System.out.println( "move disk " + n + " from " + fromTower + " to " + toTower);
            moveDISKs(n-1, auxTower, toTower, fromTower);
        }
    }
}