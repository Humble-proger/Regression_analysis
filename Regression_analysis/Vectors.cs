
using System.Numerics;
using System.Runtime.CompilerServices;

class Vectors {
    public (int, int) Shape;
    public int Size { get => Shape.Item1 * Shape.Item2; }
    private double[][] values;
    
    public Vectors(double[][] values)
    {
        this.values = values;
        this.Shape = (values.Length, values[0].Length);
    }
    private static Vectors InitVectors((int, int) shape) {
        double[][] result = new double[shape.Item1][];
        for (int i = 0; i < shape.Item2; i++)
            result[i] = new double[shape.Item2];
        return new Vectors(result);
    }
    public static Vectors Zeros((int, int) shape) {
        double[][] temp_val = new double[shape.Item1][];
        for (int i = 0; i < shape.Item1; i++) {
            temp_val[i] = new double[shape.Item2];
            for (int j = 0; j < shape.Item2; j++)
                temp_val[i][j] = 0.0;
        }
        return new Vectors(temp_val);
    }
    public static Vectors Ones((int, int) shape)
    {
        double[][] temp_val = new double[shape.Item1][];
        for (int i = 0; i < shape.Item1; i++)
        {
            temp_val[i] = new double[shape.Item2];
            for (int j = 0; j < shape.Item2; j++)
                temp_val[i][j] = 1.0;
        }
        return new Vectors(temp_val);
    }
    public static Vectors Eig((int, int) shape)
    {
        Vectors result = Vectors.Zeros(shape);
        if (shape.Item1 >= shape.Item2)
            for (int i = 0; i < shape.Item2; i++)
                result.values[i][i] = 1.0;
        else
            for (int i = 0; i < shape.Item1; i++)
                result.values[i][i] = 1.0;
        return result;
    }

    public Vectors Copy() {
        Vectors result = Vectors.InitVectors(this.Shape);
        for (int i = 0; i < this.Shape.Item1; i++)
            for (int j = 0; j < this.Shape.Item2; j++)
                result.values[i][j] = this.values[i][j];
        return result;
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
    public static Vectors Multipty(Vectors vec, double val) {
        if (double.Abs(val) < double.Epsilon) return Vectors.Zeros(vec.Shape);
        else {
            Vectors result = Vectors.InitVectors(vec.Shape);
            for (int i = 0; i < vec.Shape.Item1; i++)
                for (int j = 0; j < vec.Shape.Item2; j++)
                    result.values[i][j] = vec.values[i][j] * val;
            return result;
        }
    }
    public static Vectors Multipty(Vectors vec, Vectors val)
    {
        if (vec.Shape == val.Shape)
        {
            Vectors result = Vectors.InitVectors(vec.Shape);
            for (int i = 0; i < vec.Shape.Item1; i++)
                for (int j = 0; j < vec.Shape.Item2; j++)
                    result.values[i][j] = vec.values[i][j] * val.values[i][j];
            return result;
        }
        else throw new Exception($"The shapes of the vectors do not match. The shape of the first vector is {vec.Shape}. The shape of the second vector is {val.Shape}.");
    }
    public static Vectors Add(Vectors vec, double val)
    {
        Vectors result = vec.Copy();
        if (double.Abs(val) >= double.Epsilon)
        {
            for (int i = 0; i < vec.Shape.Item1; i++)
                for (int j = 0; j < vec.Shape.Item2; j++)
                    result.values[i][j] += val;
        }
        return result;
    }
    public static Vectors Add(Vectors vec, Vectors val)
    {
        if (vec.Shape == val.Shape)
        {
            Vectors result = Vectors.InitVectors(vec.Shape);
            for (int i = 0; i < vec.Shape.Item1; i++)
                for (int j = 0; j < vec.Shape.Item2; j++)
                    result.values[i][j] = vec.values[i][j] + val.values[i][j];
            return result;
        }
        else throw new Exception($"The shapes of the vectors do not match. The shape of the first vector is {vec.Shape}. The shape of the second vector is {val.Shape}.");
    }
    public static Vectors Sub(Vectors vec, double val) {
        Vectors result = vec.Copy();
        if (double.Abs(val) >= double.Epsilon) {
            for (int i = 0; i < vec.Shape.Item1; i++)
                for (int j = 0; j < vec.Shape.Item2; j++)
                    result.values[i][j] -= val;
        }
        return result;
    }
    public static Vectors Sub(Vectors vec, Vectors val)
    {
        if (vec.Shape == val.Shape)
        {
            Vectors result = Vectors.InitVectors(vec.Shape);
            for (int i = 0; i < vec.Shape.Item1; i++)
                for (int j = 0; j < vec.Shape.Item2; j++)
                    result.values[i][j] = vec.values[i][j] - val.values[i][j];
            return result;
        }
        else throw new Exception($"The shapes of the vectors do not match. The shape of the first vector is {vec.Shape}. The shape of the second vector is {val.Shape}.");
    }
    public static Vectors Divade(Vectors vec, double val)
    {
        if (double.Abs(val) >= double.Epsilon)
        {
            Vectors result = vec.Copy();
            for (int i = 0; i < vec.Shape.Item1; i++)
                for (int j = 0; j < vec.Shape.Item2; j++)
                    result.values[i][j] = vec.values[i][j] / val;
            return result;
        }
        else throw new Exception("Divade by Zero");
    }
    public static Vectors Divade(Vectors vec, Vectors val)
    {
        if (vec.Shape == val.Shape)
        {
            var result = Vectors.InitVectors(vec.Shape);
            for (int i = 0; i < vec.Shape.Item1; i++)
                for (int j = 0; j < vec.Shape.Item2; j++) {
                    if (double.Abs(val.values[i][j]) < double.Epsilon) throw new Exception("Divade by Zero");
                    result.values[i][j] = vec.values[i][j] / val.values[i][j];
                }
            return result;
        }
        else throw new Exception($"The shapes of the vectors do not match. The shape of the first vector is {vec.Shape}. The shape of the second vector is {val.Shape}.");
    }
    public Vectors Dot(Vectors vec) {
        if (this.Shape.Item2 == vec.Shape.Item1)
        {
            Vectors result_vec = Vectors.Zeros((this.Shape.Item1, vec.Shape.Item2));
            double sumator = 0.0;
            for (int i = 0; i < this.Shape.Item1; i++)
                for (int j = 0; j < vec.Shape.Item2; j++)
                {
                    sumator = 0;
                    for (int k = 0; k < this.Shape.Item2; k++)
                        sumator += this.values[i][k] * vec.values[k][j];
                    result_vec.values[i][j] = sumator;
                }
            return result_vec;
        }
        else throw new Exception($"Vector multiplication requires the shape to be (n, m) -> (m, k). Currently it is {this.Shape} -> {vec.Shape}.");
    }
    public Vectors Square() {
        if (this.Shape.Item1 == this.Shape.Item2) {
            Vectors result_vec = Vectors.Zeros(this.Shape);
            double sumator = 0.0;
            for (int i = 0; i < this.Shape.Item1; i++)
                for (int j = 0; j < this.Shape.Item2; j++)
                {
                    sumator = 0;
                    for (int k = 0; k < this.Shape.Item2; k++)
                        sumator += this.values[i][k] * this.values[k][j];
                    result_vec.values[i][j] = sumator;
                }
            return result_vec;
        }
        else throw new Exception($"Vector multiplication requires the shape to be (n, m) -> (m, k). Currently it is {this.Shape} -> {this.Shape}.");
    }
    public static double Max(Vectors vec) {
        double result = 0;
        for (int i = 0; i < vec.Shape.Item1; i++)
            for (int j = 0; j < vec.Shape.Item2; j++) {
                var tmp = double.Abs(vec.values[i][j]);
                if (tmp > result)
                    result = tmp;
            }
        return result;
    }
    public static double Min(Vectors vec)
    {
        double result = double.MaxValue;
        for (int i = 0; i < vec.Shape.Item1; i++)
            for (int j = 0; j < vec.Shape.Item2; j++)
            {
                var tmp = double.Abs(vec.values[i][j]);
                if (tmp < result)
                    result = tmp;
            }
        return result;
    }

    public Vectors Inv(int maxIterations = 100, double tolerance = 1e-10) {
        if (this.Shape.Item1 == this.Shape.Item2)
        {
            Vectors identity = Vectors.Eig(this.Shape);
            Vectors inverse = Vectors.Eig(this.Shape); // Начальное приближение (единичная матрица)

            // Инициализация начального приближения
            Vectors currentApproximation = Vectors.Eig(this.Shape);

            for (int k = 0; k < maxIterations; k++)
            {
                // Вычисляем произведение текущего приближения и (2I - A * текущий приближенец)
                var product = this.Dot(currentApproximation);
                var nextApproximation = Vectors.Sub(identity, product);
                nextApproximation = currentApproximation.Dot(nextApproximation);
                // Проверка на сходимость
                if (Vectors.Max(Vectors.Sub(currentApproximation, nextApproximation)) < tolerance)
                    return nextApproximation;
                currentApproximation = nextApproximation;
            }

            throw new InvalidOperationException("Could not find the inverse matrix in the given number of iterations.");
        }
        else throw new Exception("The Vector in not square. Cannon be reversed.");
    }
    private static double MaxDifference(double[,] a, double[,] b)
    {
        double maxDiff = 0.0;
        int rows = a.GetLength(0);
        int cols = a.GetLength(1);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double diff = Math.Abs(a[i, j] - b[i, j]);
                if (diff > maxDiff)
                {
                    maxDiff = diff;
                }
            }
        }

        return maxDiff;
    }
}


class TestClass
{
    static void Main(string[] args)
    {
        Vectors vector1 = new Vectors( [ [1, 2, 3], [2, 3, 4], [3, 4, 8] ] );
        vector1 = vector1.Inv();
        Console.WriteLine(vector1);
    }
}




/*
private static double[,] CreateIdentityMatrix(int size)
{
    double[,] identity = new double[size, size];
    for (int i = 0; i < size; i++)
    {
        identity[i, i] = 1.0;
    }
    return identity;
}

private static double[,] Multiply(double[,] a, double[,] b)
{
    int aRows = a.GetLength(0);
    int aCols = a.GetLength(1);
    int bCols = b.GetLength(1);
    double[,] result = new double[aRows, bCols];

    for (int i = 0; i < aRows; i++)
    {
        for (int j = 0; j < bCols; j++)
        {
            result[i, j] = 0;
            for (int k = 0; k < aCols; k++)
            {
                result[i, j] += a[i, k] * b[k, j];
            }
        }
    }

    return result;
}

private static double[,] Subtract(double[,] a, double[,] b)
{
    int rows = a.GetLength(0);
    int cols = a.GetLength(1);
    double[,] result = new double[rows, cols];

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[i, j] = a[i, j] - b[i, j];
        }
    }

    return result;
}

private static double MaxDifference(double[,] a, double[,] b)
{
    double maxDiff = 0.0;
    int rows = a.GetLength(0);
    int cols = a.GetLength(1);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double diff = Math.Abs(a[i, j] - b[i, j]);
            if (diff > maxDiff)
            {
                maxDiff = diff;
            }
        }
    }

    return maxDiff;
}
*/