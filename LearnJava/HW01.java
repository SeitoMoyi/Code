import java.util.Scanner;

public class HW01 {
    static int[] a = new int[100];
    public static void main(String[] args){
        System.out.println("请输入n和r，以空格分隔：");
        int n, k;
        Scanner scan = new Scanner(System.in);
        n = scan.nextInt();
        k = scan.nextInt();
        scan.close();
        a[0] = k;
        choose(n,k);
    }
    static void choose(int n, int k){
        for (int i = n; i >= k; i--){
            a[k] = i;
            if (k > 1){
                choose(i - 1, k - 1);
            }
            else{
                System.out.print('|');
                for (int j = a[0]; j >= 1 ; j--){
                    System.out.print(a[j]);
                    System.out.print("\t|");
                }
                System.out.println();
            }
        }
    }
}