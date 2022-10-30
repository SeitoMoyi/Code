public class TowersOfHanoi {
    public static void main(String[] args) {
        int sum = 0;
        int n = 100;
        for(int i = 0; i < n; i++ )
            for(int j = 0; j <i*i; j++ )
                for(int k = 0; k <j; k++ )
                    sum++;
        System.out.println(sum);
    }
}