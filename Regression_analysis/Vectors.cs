
using System.Numerics;

class Vector2 {
    public (int, int) Shape;
    public int Size { get => Shape.Item1 * Shape.Item2; }
    private double[][] values;
    
    public Vector2(double[][] values)
    {
        this.values = values;
        this.Shape = (values.Length, values[0].Length);
    }
    public static Vector2 Zeros((int, int) shape) {
        double[][] temp_val = new double[shape.Item1][];
        for (int i = 0; i < shape.Item1; i++) {
            temp_val[i] = new double[shape.Item2];
            for (int j = 0; j < shape.Item2; j++)
                temp_val[i][j] = 0.0;
        }
        return new Vector2(temp_val);
    }

    public void T() {
        double[][] new_values = new double[this.Shape.Item2][];
        for (int j = 0; j < this.Shape.Item2; j++)
        {
            new_values[j] = new double[this.Shape.Item1];
            new_values[j][0] = this.values[0][j];
        }
        for (int i = 1; i < this.Shape.Item1; i++) {
            for (int j = 0; j < this.Shape.Item2; j++) {
                new_values[j][i] = this.values[i][j];                
            }
        }
        this.values = new_values;
        (this.Shape.Item1, this.Shape.Item2) = (this.Shape.Item2, this.Shape.Item1);
    }
    public void MultiptyScalar(double val) {
        for (int i = 0; i < this.Shape.Item1; i++)
            for (int j = 0; j < this.Shape.Item2; j++)
                this.values[i][j] *= val;
    }
}


class TestClass
{
    static void Main(string[] args)
    {
        Vector2 vector1 = new Vector2( [ [1, 2, 3] ] );
        vector1.T();
        Console.WriteLine(vector1);
    }
}

