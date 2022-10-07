import java.util.Scanner;

public class homework{
    public static void main(String[] args){
        Scanner scan = new Scanner(System.in);
        int n = scan.nextInt();
        scan.close();
        int[] list = new int[20];
        for (int i = 0; i < n;i++ ){
            list[i] = i+1;
        }
        arrange(list, 0, n-1);
    }
    public static void swap(int[] list,int x, int y){
        int t = list[x];
        list[x] = list[y];
        list[y] = t;
    }
    public static void arrange(int[] list,int m,int n){
        if (m == n){
            for(int i = 0; i <= n; i++){
                System.out.print(list[i]);
            }
            System.out.println("");
            return;
        }
        else{
            for(int i = m; i <= n; i++){
            swap(list, m, i);
            arrange(list, m+1, n);
            swap(list, m, i);
            }
        }
    }
}
