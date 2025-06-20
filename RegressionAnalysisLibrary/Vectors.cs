﻿using System.Text;

namespace RegressionAnalysisLibrary
{
    public class Vectors
    {
        public (int, int) Shape;
        public int Size { get => Shape.Item1 * Shape.Item2; }
        private readonly double[][] _values;
        private bool _transposes = false;

        public Vectors(double[][] value)
        {
            _values = value;
            Shape = (value.Length, value[0].Length);
        }
        public Vectors(double[] value)
        {
            _values = [value];
            Shape = (1, value.Length);
        }

        public static Vectors InitVectors((int, int) shape)
        {
            var result = new double[shape.Item1][];
            for (var i = 0; i < shape.Item1; i++)
                result[i] = new double[shape.Item2];
            return new Vectors(result);
        }

        public static Vectors RoundVector(Vectors vector, int roundDecimals)
        {
            for (var i = 0; i < vector.Shape.Item1; i++)
                for (var j = 0; j < vector.Shape.Item2; j++)
                    vector[i, j] = double.Round(vector[i, j], roundDecimals);
            return vector;
        }

        public static Vectors Zeros((int, int) shape)
        {
            var temp_val = new double[shape.Item1][];
            for (var i = 0; i < shape.Item1; i++)
            {
                temp_val[i] = new double[shape.Item2];
                for (var j = 0; j < shape.Item2; j++)
                    temp_val[i][j] = 0.0;
            }
            return new Vectors(temp_val);
        }
        public static Vectors Ones((int, int) shape)
        {
            var temp_val = new double[shape.Item1][];
            for (var i = 0; i < shape.Item1; i++)
            {
                temp_val[i] = new double[shape.Item2];
                for (var j = 0; j < shape.Item2; j++)
                    temp_val[i][j] = 1.0;
            }
            return new Vectors(temp_val);
        }
        public static Vectors Eig((int, int) shape, double value = 1.0)
        {
            var result = new double[shape.Item1][];
            for (var i = 0; i < shape.Item1; i++)
            {
                result[i] = new double[shape.Item2];
                for (var j = 0; j < i; j++)
                    result[i][j] = 0.0;
                result[i][i] = value;
                for (var k = i + 1; k < shape.Item2; k++)
                    result[i][k] = 0.0;
            }
            return new Vectors(result);
        }

        public double Mean() => Sum(this) / Size;
        public static double Mean(Vectors vector) => Sum(vector) / vector.Size;

        public double Variance(double? mean = null)
        {
            var variance = 0.0;
            if (mean is null)
                for (var i = 0; i < Shape.Item1; i++)
                    for (var j = 0; j < Shape.Item2; j++)
                        variance += double.Pow(this[i, j], 2);
            else
                for (var i = 0; i < Shape.Item1; i++)
                    for (var j = 0; j < Shape.Item2; j++)
                        variance += double.Pow(this[i, j] - (double) mean, 2);
            return variance / Size;
        }

        public static double Variance(Vectors vectors, double? mean = null)
        {
            var variance = 0.0;
            if (mean is null)
                for (var i = 0; i < vectors.Shape.Item1; i++)
                    for (var j = 0; j < vectors.Shape.Item2; j++)
                        variance += double.Pow(vectors[i, j], 2);
            else
                for (var i = 0; i < vectors.Shape.Item1; i++)
                    for (var j = 0; j < vectors.Shape.Item2; j++)
                        variance += double.Pow(vectors[i, j] - (double) mean, 2);
            return variance / vectors.Size;
        }
        public double VarianceNoOffset(double? mean = null)
        {
            mean ??= Mean();
            var variance = 0.0;
            for (var i = 0; i < Shape.Item1; i++)
                for (var j = 0; j < Shape.Item2; j++)
                    variance += double.Pow(this[i, j] - (double) mean, 2);
            return variance / (Size - 1);
        }

        public static double VarianceNoOffset(Vectors vectors, double? mean = null)
        {
            mean ??= Mean(vectors);
            var variance = 0.0;
            for (var i = 0; i < vectors.Shape.Item1; i++)
                for (var j = 0; j < vectors.Shape.Item2; j++)
                    variance += double.Pow(vectors[i, j] - (double) mean, 2);
            return variance / (vectors.Size - 1);
        }

        public void Sort()
        {
            if (_transposes)
            {
                T();
                for (var i = 0; i < Shape.Item1; i++)
                    _values[i] = [.. _values[i].OrderBy(x => x)];
                T();
            }
            else
                for (var i = 0; i < Shape.Item1; i++)
                    _values[i] = [.. _values[i].OrderBy(x => x)];
        }

        public static double Median(Vectors vector, bool sort = false)
        {
            if (!vector.IsVector() || vector.Size == 0) throw new ArgumentException("Массив данных должен быть вектором.");
            if (vector.Size == 1) return vector[0];
            if (!sort)
            {
                vector = vector.Copy();
                vector.Sort();
            }
            return vector.Size + 1 % 2 != 0 ? vector[(vector.Size + 1) / 2] : (vector[vector.Size / 2] + vector[vector.Size / 2 + 1]) / 2;
        }

        public static double Percentile(Vectors vector, double procentile, bool sort = false)
        {
            if (!vector.IsVector()) throw new ArgumentException("Массив данных должен быть вектором.");
            if (vector.Size == 0) throw new ArgumentException("Массив должен быть не пустым.");
            if (procentile < 0 || procentile > 1) throw new ArgumentException("Введённый проценталь выходит за диапозон определения [0, 1]");
            if (!sort)
            {
                vector = vector.Copy();
                vector.Sort();
            }
            var index = (vector.Size - 1) * procentile;
            var lowerIndex = (int) Math.Floor(index);
            var fraction = index - lowerIndex;
            return lowerIndex >= vector.Size - 1 ? vector[-1] : vector[lowerIndex] + fraction * (vector[lowerIndex + 1] - vector[lowerIndex]);
        }
        public Vectors Copy()
        {
            var result = new double[Shape.Item1][];
            for (var i = 0; i < Shape.Item1; i++)
            {
                result[i] = new double[Shape.Item2];
                for (var j = 0; j < Shape.Item2; j++)
                    result[i][j] = this[i, j];
            }
            return new Vectors(result);
        }

        public static int MinIndex(Vectors vec)
        {
            var min = double.MaxValue;
            var index = 0;

            for (var i = 0; i < vec.Shape.Item2; i++)
                if (vec[i] < min)
                {
                    min = vec[i];
                    index = i;
                }

            return index;
        }

        public static int MaxIndex(Vectors vec)
        {
            var max = double.MinValue;
            var index = 0;

            for (var i = 0; i < vec.Shape.Item2; i++)
                if (vec[i] > max)
                {
                    max = vec[i];
                    index = i;
                }

            return index;
        }

        public static int SecondMaxIndex(Vectors vec, int excludeIndex)
        {
            var max = double.MinValue;
            var index = 0;

            for (var i = 0; i < vec.Shape.Item2; i++)
                if (i != excludeIndex && vec[i] > max)
                {
                    max = vec[i];
                    index = i;
                }

            return index;
        }


        public void ShaffleRows(Random rand)
        {
            int n = Shape.Item1, k;
            while (n > 1)
            {
                n--;
                k = rand.Next(n + 1);
                if (_transposes)
                    for (var i = 0; i < Shape.Item2; i++)
                        (_values[i][n], _values[i][k]) = (_values[i][k], _values[i][n]);
                else
                    (_values[n], _values[k]) = (_values[k], _values[n]);
            }
        }

        public static double NormDifference(Vectors vec, double minValue)
        {
            double sum = 0;
            for (var i = 0; i < vec.Shape.Item2; i++)
                sum += Math.Pow(vec[i] - minValue, 2);
            return Math.Sqrt(sum);
        }

        public static bool Equals(Vectors v1, Vectors v2)
        {
            if (v1.Shape.Item2 != v2.Shape.Item2)
                return false;

            for (var i = 0; i < v1.Shape.Item2; i++)
                if (Math.Abs(v1[i] - v2[i]) > double.Epsilon)
                    return false;

            return true;
        }


        public Vectors T()
        {
            var result = Copy();
            result._transposes = !result._transposes;
            (result.Shape.Item1, result.Shape.Item2) = (result.Shape.Item2, result.Shape.Item1);
            return result;
        }
        public static Vectors Multipty(Vectors vec, double val)
        {
            if (double.Abs(val) < double.Epsilon) return Zeros(vec.Shape);
            else
            {
                var result = new double[vec.Shape.Item1][];
                for (var i = 0; i < vec.Shape.Item1; i++)
                {
                    result[i] = new double[vec.Shape.Item2];
                    for (var j = 0; j < vec.Shape.Item2; j++)
                        result[i][j] = vec[i, j] * val;
                }
                return new Vectors(result);
            }
        }
        public static Vectors Multipty(Vectors vec, Vectors val)
        {
            if (vec.Shape == val.Shape)
            {
                var result = new double[vec.Shape.Item1][];
                for (var i = 0; i < vec.Shape.Item1; i++)
                {
                    result[i] = new double[vec.Shape.Item2];
                    for (var j = 0; j < vec.Shape.Item2; j++)
                        result[i][j] = vec[i, j] * val[i, j];
                }
                return new Vectors(result);
            }
            else throw new Exception($"The shapes of the vectors do not match. The shape of the first vector is {vec.Shape}. The shape of the second vector is {val.Shape}.");
        }
        public static Vectors Add(Vectors vec, double val)
        {
            if (double.Abs(val) >= double.Epsilon)
            {
                var result = new double[vec.Shape.Item1][];
                for (var i = 0; i < vec.Shape.Item1; i++)
                {
                    result[i] = new double[vec.Shape.Item2];
                    for (var j = 0; j < vec.Shape.Item2; j++)
                        result[i][j] = vec[i, j] + val;
                }
                return new Vectors(result);
            }
            return vec;
        }
        public static Vectors Add(Vectors vec, Vectors val)
        {
            if (vec.Shape == val.Shape)
            {
                var result = new double[vec.Shape.Item1][];
                for (var i = 0; i < vec.Shape.Item1; i++)
                {
                    result[i] = new double[vec.Shape.Item2];
                    for (var j = 0; j < vec.Shape.Item2; j++)
                        result[i][j] = vec[i, j] + val[i, j];
                }
                return new Vectors(result);
            }
            else throw new Exception($"The shapes of the vectors do not match. The shape of the first vector is {vec.Shape}. The shape of the second vector is {val.Shape}.");
        }
        public static Vectors Sub(Vectors vec, double val)
        {
            if (double.Abs(val) >= double.Epsilon)
            {
                var result = new double[vec.Shape.Item1][];
                for (var i = 0; i < vec.Shape.Item1; i++)
                {
                    result[i] = new double[vec.Shape.Item2];
                    for (var j = 0; j < vec.Shape.Item2; j++)
                        result[i][j] = vec[i, j] - val;
                }
                return new Vectors(result);
            }
            return vec;
        }
        public static Vectors Sub(Vectors vec, Vectors val)
        {
            if (vec.Shape == val.Shape)
            {
                var result = new double[vec.Shape.Item1][];
                for (var i = 0; i < vec.Shape.Item1; i++)
                {
                    result[i] = new double[vec.Shape.Item2];
                    for (var j = 0; j < vec.Shape.Item2; j++)
                        result[i][j] = vec[i, j] - val[i, j];
                }
                return new Vectors(result);
            }
            else throw new Exception($"The shapes of the vectors do not match. The shape of the first vector is {vec.Shape}. The shape of the second vector is {val.Shape}.");
        }
        public static Vectors Divade(Vectors vec, double val)
        {
            if (double.Abs(val) >= double.Epsilon)
            {
                var result = new double[vec.Shape.Item1][];
                for (var i = 0; i < vec.Shape.Item1; i++)
                {
                    result[i] = new double[vec.Shape.Item2];
                    for (var j = 0; j < vec.Shape.Item2; j++)
                        result[i][j] = vec[i, j] / val;
                }
                return new Vectors(result);
            }
            else throw new Exception("Divade by Zero");
        }
        public static Vectors Divade(Vectors vec, Vectors val)
        {
            if (vec.Shape == val.Shape)
            {
                var result = new double[vec.Shape.Item1][];
                for (var i = 0; i < vec.Shape.Item1; i++)
                {
                    result[i] = new double[vec.Shape.Item2];
                    for (var j = 0; j < vec.Shape.Item2; j++)
                    {
                        if (double.Abs(val[i, j]) < double.Epsilon) throw new Exception("Divade by Zero");
                        result[i][j] = vec[i, j] / val[i, j];
                    }
                }
                return new Vectors(result);
            }
            else throw new Exception($"The shapes of the vectors do not match. The shape of the first vector is {vec.Shape}. The shape of the second vector is {val.Shape}.");
        }
        public Vectors Dot(Vectors vec)
        {
            if (Shape.Item2 != vec.Shape.Item1)
                throw new Exception($"Vector multiplication requires the shape to be (n, m) -> (m, k). Currently it is {Shape} -> {vec.Shape}.");

            var result = new double[Shape.Item1][];
            for (var i = 0; i < Shape.Item1; i++)
            {
                result[i] = new double[vec.Shape.Item2];
                for (var j = 0; j < vec.Shape.Item2; j++)
                {
                    result[i][j] = 0.0;
                    for (var k = 0; k < Shape.Item2; k++)
                        result[i][j] += this[i, k] * vec[k, j];
                }
            }
            return new Vectors(result);
        }

        public static double Max(Vectors vec)
        {
            var result = double.MinValue;
            for (var i = 0; i < vec.Shape.Item1; i++)
                for (var j = 0; j < vec.Shape.Item2; j++)
                {
                    var tmp = vec[i, j];
                    if (tmp > result)
                        result = tmp;
                }
            return result;
        }
        public static double Min(Vectors vec)
        {
            var result = double.MaxValue;
            for (var i = 0; i < vec.Shape.Item1; i++)
                for (var j = 0; j < vec.Shape.Item2; j++)
                {
                    var tmp = vec[i, j];
                    if (tmp < result)
                        result = tmp;
                }
            return result;
        }
        public static double Norm(Vectors vec)
        {
            if (vec.IsVector())
            {
                var result = vec.Dot(vec.T())[0];
                return Math.Sqrt(result);
            }
            throw new Exception("The argument must be a vector.");
        }

        public static Vectors Inv(Vectors a)
        {
            if (a.Shape.Item1 == a.Shape.Item2)
            {
                var l = InitVectors(a.Shape);
                var u = InitVectors(a.Shape);
                // LU-разложение
                for (var i = 0; i < a.Shape.Item1; i++)
                {
                    for (var j = i; j < a.Shape.Item2; j++)
                    {
                        u._values[i][j] = a._values[i][j];
                        for (var k = 0; k < i; k++)
                            u._values[i][j] -= l._values[i][k] * u._values[k][j];
                    }
                    for (var j = i; j < a.Shape.Item1; j++)
                        if (i == j)
                            l._values[i][i] = 1; // диагональные элементы равны 1
                        else
                        {
                            l._values[j][i] = a._values[j][i];
                            for (var k = 0; k < i; k++)
                                l._values[j][i] -= l._values[j][k] * u._values[k][i];
                            l._values[j][i] /= u._values[i][i];
                        }
                }

                // Теперь решим систему уравнений для получения обратной матрицы
                var a_inv = InitVectors(a.Shape);

                for (var i = 0; i < a.Shape.Item1; i++)
                {
                    // Решаем Ly = e_i для y
                    var y = new double[a.Shape.Item1];
                    y[i] = 1;
                    for (var j = 0; j < a.Shape.Item2; j++)
                        for (var k = 0; k < j; k++)
                            y[j] -= l._values[j][k] * y[k];

                    // Теперь решаем Ux = y для x
                    for (var j = a.Shape.Item2 - 1; j >= 0; j--)
                    {
                        a_inv._values[j][i] = y[j];
                        for (var k = j + 1; k < a.Shape.Item1; k++)
                            a_inv._values[j][i] -= u._values[j][k] * a_inv._values[k][i];
                        a_inv._values[j][i] /= u._values[j][j];
                    }
                }

                return a_inv;
            }
            else throw new Exception("The Vector in not square. Cannon be reversed.");
        }

        public static Vectors GetColumn(Vectors vector, int colIndex)
        {
            var result = new double[vector.Shape.Item1];
            colIndex = vector.GetIndex(colIndex, false);
            for (var i = 0; i < vector.Shape.Item1; i++)
                result[i] = vector[i, colIndex];
            return new Vectors(result);
        }

        // Обновляет столбец в матрице D
        public static void SetColumn(Vectors vector, Vectors column, int colIndex)
        {
            for (var i = 0; i < vector.Shape.Item1; i++)
                vector[i, colIndex] = column[i];
        }

        public static Vectors GetRow(Vectors vector, int index)
        {
            var result = new double[vector.Shape.Item2];
            index = vector.GetIndex(index, false);
            if (vector._transposes)
                for (var j = 0; j < vector.Shape.Item2; j++)
                    result[j] = vector._values[j][index];
            else
                for (var i = 0; i < vector.Shape.Item2; i++)
                    result[i] = vector._values[index][i];
            return new Vectors(result);
        }
        public void SetRow(Vectors row, int index)
        {
            if (!row.IsVector())
                throw new Exception("row is not Vector");
            if (row.Size != Shape.Item2)
                throw new Exception("The 'row' size does not match the vector row length");
            index = GetIndex(index, false);
            if (_transposes)
                for (var j = 0; j < Shape.Item2; j++)
                    _values[j][index] = row[j];
            else
                for (var i = 0; i < Shape.Item2; i++)
                    _values[index][i] = row[i];
        }
        private int GetIndex(int index, bool item2 = true)
        {
            return item2
                ? int.Abs(index) > Size ? throw new Exception("Index out of range") : index < 0 ? index + Shape.Item2 : index
                : int.Abs(index) > Size ? throw new Exception("Index out if range.") : index < 0 ? index + Shape.Item1 : index;
        }
        public bool IsVector()
        {
            return Shape.Item1 == 1 || Shape.Item2 == 1;
        }

        public static double Sum(Vectors vector)
        {
            var sum = 0.0;
            var compensation = 0.0; // Коррекция ошибки
            for (var i = 0; i < vector.Shape.Item1; i++)
                for (var j = 0; j < vector.Shape.Item2; j++)
                {
                    var y = vector[i, j] - compensation;
                    var t = sum + y;
                    compensation = t - sum - y;
                    sum = t;
                }
            return sum;
        }

        public double[] ToArrayVector()
        {
            if (IsVector())
            {
                var result = new double[Size];
                for (var i = 0; i < Size; i++)
                    result[i] = this[i];
                return result;
            }
            throw new Exception("Объект не является вектором");
        }

        public double[][] ToArray()
        {
            return (double[][]) _values.Clone();
        }

        public Vectors Clone()
        {
            var result = InitVectors(Shape);
            for (var i = 0; i < Shape.Item1; i++)
                for (var j = 0; j < Shape.Item2; j++)
                    result[i, j] = this[i, j];
            return result;
        }

        public override string? ToString()
        {
            if (Size > 0)
            {
                StringBuilder sb = new();
                for (var i = 0; i < Shape.Item1; i++)
                {
                    sb.Append($"[ {this[i, 0].ToString("E2")}");
                    for (var j = 1; j < Shape.Item2; j++)
                        sb.Append($" {this[i, j].ToString("E2")}");
                    sb.Append(" ]\n");
                }
                return sb.ToString();
            }
            else return "Empty";
        }
        public bool SaveToDAT(string path, string title = "Temp title")
        {
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
            var stream = new StringBuilder(title);
            stream.Append('\n');
            stream.AppendLine($"0 {Size}");
            if (IsVector())
            {
                var index = 0;
                for (var i = 0; i < Size / 5; i++)
                {
                    for (var j = 0; j < 4 && index < Size; j++, index++)
                    {
                        stream.Append(FormattableString.Invariant($"{this[index]}"));
                        stream.Append(' ');
                    }
                    if (index < Size)
                    {
                        stream.Append(FormattableString.Invariant($"{this[index]}"));
                        index++;
                    }
                    stream.Append('\n');
                }
            }
            else
                for (var i = 0; i < Shape.Item1; i++)
                {
                    for (var j = 0; j < Shape.Item2; j++)
                    {
                        stream.Append(FormattableString.Invariant($"{this[i, j]}"));
                        stream.Append(' ');
                    }
                    stream.Append('\n');
                }
            File.WriteAllText(path, stream.ToString(), Encoding.GetEncoding(1251));
            return File.Exists(path);
        }
        public double this[int i, int j]
        {
            get
            {
                int tmp = GetIndex(i, false), tmp_2 = GetIndex(j);
                return _transposes ? _values[tmp_2][tmp] : _values[tmp][tmp_2];
            }
            set
            {
                int tmp = GetIndex(i, false), tmp_2 = GetIndex(j);
                if (_transposes)
                    _values[tmp_2][tmp] = value;
                else
                    _values[tmp][tmp_2] = value;
            }
        }
        public double this[int i]
        {
            get
            {
                if (_transposes)
                {
                    var tmp = GetIndex(i, false);
                    return Shape.Item1 == 1 ? _values[tmp][0] : _values[0][tmp];
                }
                else
                {
                    var tmp = GetIndex(i);
                    return Shape.Item1 == 1 ? _values[0][tmp] : _values[tmp][0];
                }
            }
            set
            {
                if (_transposes)
                {
                    var tmp = GetIndex(i, false);
                    if (Shape.Item1 == 1)
                        _values[tmp][0] = value;
                    else
                        _values[0][tmp] = value;
                }
                else
                {
                    var tmp = GetIndex(i);
                    if (Shape.Item1 == 1)
                        _values[0][tmp] = value;
                    else
                        _values[tmp][0] = value;
                }
            }
        }

        public static Vectors operator +(Vectors s1, Vectors s2) => Add(s1, s2);
        public static Vectors operator +(Vectors s1, double s2) => Add(s1, s2);
        public static Vectors operator +(double s1, Vectors s2) => Add(s2, s1);
        public static Vectors operator -(Vectors s1, Vectors s2) => Sub(s1, s2);
        public static Vectors operator -(Vectors s1, double s2) => Sub(s1, s2);
        public static Vectors operator -(double s1, Vectors s2) => Sub(s2, s1);
        public static Vectors operator *(Vectors s1, Vectors s2) => Multipty(s1, s2);
        public static Vectors operator *(Vectors s1, double s2) => Multipty(s1, s2);
        public static Vectors operator *(double s1, Vectors s2) => Multipty(s2, s1);
        public static Vectors operator /(Vectors s1, Vectors s2) => Divade(s1, s2);
        public static Vectors operator /(Vectors s1, double s2) => Divade(s1, s2);
        public static Vectors operator /(double s1, Vectors s2) => Divade(s2, s1);
        public static Vectors operator &(Vectors s1, Vectors s2) => s1.Dot(s2);
        public static Vectors operator -(Vectors s1) => Multipty(s1, -1);
    }
}

