using System.Reflection;
using System.Runtime.CompilerServices;

namespace Regression_analysis
{
    internal static class MathConstants
    {
        public const double Sqrt2 = 1.4142135623730950;
        public const double Sqrt3 = 1.7320508075688773;
    }

    public enum TypeDisribution {
        Uniform,
        Normal,
        Exponential,
        Laplace,
        Cauchy,
        Gamma
    }

    [AttributeUsage(AttributeTargets.Class, Inherited = false, AllowMultiple = false)]
    public class DistributionNameAttribute : Attribute
    {
        public string Name { get; }
        public TypeDisribution Type { get; }

        public DistributionNameAttribute(string name, TypeDisribution type)
        {
            Name = name;
            Type = type;
        }
    }

    public interface IRandomDistribution
    {
        public string Name { get; }
        public int CountParametrsDistribution { get; }
        public string[] NameParameters { get; }
        public (double?, double?)[]? BoundsParameters { get; }
        Vectors DefaultParametrs { get; }
        public double Generate();
        public double? Generate(Vectors paramsDist);
        public Vectors? Generate((int, int) shape, Vectors paramsDist);
        public bool CheckParamsDist(Vectors paramsDist);
    }

    public class DistributionFactory
    {
        private readonly Dictionary<TypeDisribution, IRandomDistribution> _distributions;

        public DistributionFactory(int? seed = null)
        {
            _distributions = [];
            LoadDistributions(seed);
        }

        private void LoadDistributions(int? seed)
        {
            var assembly = Assembly.GetExecutingAssembly();

            foreach (var type in assembly.GetTypes()
                .Where(t => typeof(IRandomDistribution).IsAssignableFrom(t)
                        && !t.IsInterface
                        && !t.IsAbstract))
            {
                var attr = type.GetCustomAttribute<DistributionNameAttribute>();
                if (attr == null) continue;

                try
                {
                    var instance = Activator.CreateInstance(type, seed) as IRandomDistribution;
#pragma warning disable CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
                    _distributions[attr.Type] = instance;
#pragma warning restore CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to create {attr.Name}: {ex.Message}");
                }
            }
        }

        public IRandomDistribution GetDistribution(TypeDisribution type)
        {
            return _distributions.TryGetValue(type, out var distribution)
                ? distribution
                : throw new KeyNotFoundException($"Distribution {type} not found");
        }

        public IEnumerable<(TypeDisribution Type, string Name, IRandomDistribution Instance)> GetAllDistributions()
        {
            return _distributions.Select(kv => (kv.Key, GetDistributionName(kv.Value.GetType()), kv.Value));
        }

        private static string GetDistributionName(Type distributionType) => distributionType.GetCustomAttribute<DistributionNameAttribute>()?.Name
                   ?? distributionType.Name;
    }
    [DistributionName("Равномерное распределение", TypeDisribution.Uniform)]
    public class UniformDistribution(int? seed = null) : IRandomDistribution
    {
        public int CountParametrsDistribution => 2;
        public Vectors DefaultParametrs => new([0, 1]);

        public string Name => "Равномерное";

        private readonly Random _random = seed is null ? new Random() : new Random((int) seed);

        private static bool CheckParams(Vectors paramsDist)
        {
            return paramsDist.Size == 2 && paramsDist[0] < paramsDist[1];
        }
        public bool CheckParamsDist(Vectors paramsDist) => CheckParams(paramsDist);

        private static double Uniform(double a, double b, in Random rand) => a + rand.NextDouble() * (b - a);
        public static double Generate(in Random rand) => rand.NextDouble();
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParams(paramsDist) ? Uniform(paramsDist[0], paramsDist[1], rand) : null;
        public double Generate() => _random.NextDouble();
        public double? Generate(Vectors paramsDist) => this.CheckParamsDist(paramsDist) ? global::Regression_analysis.UniformDistribution.Uniform(paramsDist[0], paramsDist[1], _random) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist)
        {
            if (!this.CheckParamsDist(paramsDist)) return null;
            var vec = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    vec[i, j] = Uniform(paramsDist[0], paramsDist[1], _random);
            return vec;
        }

        public static Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand) {
            if (!CheckParams(paramsDist)) return null;
            var vec = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    vec[i, j] = Uniform(paramsDist[0], paramsDist[1], rand);
            return vec;
        }

        public static Moment Mean => (paramDist) => (paramDist[1] - paramDist[0]) / 2;
        public static Moment Var => (paramDist) => double.Pow(paramDist[1] - paramDist[0], 2) / 12;

        public string[] NameParameters => ["Нижняя граница", "Верхняя граница"];

        public (double?, double?)[]? BoundsParameters => null;
    }

    [DistributionName("Экспоненциальное распределение", TypeDisribution.Exponential)]
    public class ExponentialDistribution(int? seed = null) : IRandomDistribution {

        public int CountParametrsDistribution => 2;
        public Vectors DefaultParametrs => new([0, 1]);

        public string Name => "Экспоненциальное";

        private readonly Random _random = seed is null ? new Random() : new Random((int) seed);
        private static double Exponential(double loc, double scale, double u) => -Math.Log(1 - u) * scale + loc;
        private static bool CheckParams(Vectors paramsDist) => paramsDist.Size == 2 && paramsDist[1] > 0;
        public bool CheckParamsDist(Vectors paramsDist) => CheckParams(paramsDist);
        public double Generate() => Exponential(0, 1, _random.NextDouble());
        public double? Generate(Vectors paramsDist) => CheckParamsDist(paramsDist) ? Exponential(paramsDist[0], paramsDist[1], _random.NextDouble()) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist) {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Exponential(paramsDist[0], paramsDist[1], _random.NextDouble());
            return result;
        }
        public static double Generate(in Random rand) => Exponential(0, 1, rand.NextDouble());
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParams(paramsDist) ? Exponential(paramsDist[0], paramsDist[1], rand.NextDouble()) : null;
        public static Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand) {
            if (!CheckParams(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Exponential(paramsDist[0], paramsDist[1], rand.NextDouble());
            return result;
        }

        public static Moment Mean => (paramDist) => paramDist[1] + paramDist[0];
        public static Moment Var => (paramDist) => paramDist[1] * paramDist[1];

        public string[] NameParameters => ["Сдвиг", "Маштаб"];

        public (double?, double?)[]? BoundsParameters => [(null, null), (0, null)];
    }

    [DistributionName("Распределение Лапласа", TypeDisribution.Laplace)]
    public class LaplaceDistribution(int? seed = null) : IRandomDistribution
    {
        public int CountParametrsDistribution => 2;
        public Vectors DefaultParametrs => new([0, 1]);

        public string Name => "Лапласа";

        private readonly Random _random = seed is null ? new Random() : new Random((int) seed);
        private static double Laplace(double loc, double scale, in Random rand) {
            var u = rand.NextDouble();
            if (u < 0.5)
            {
                return loc + scale * double.Log(2 * u);
            }
            else if (u > 0.5) {
                return loc - scale * double.Log(2 * (1 - u));
            }
            return loc;
        }
        private static bool CheckParams(Vectors paramsDist) => paramsDist.Size == 2 && paramsDist[1] > 0;
        public bool CheckParamsDist(Vectors paramsDist) => CheckParams(paramsDist);
        public double Generate() => Laplace(0, 1, _random);
        public double? Generate(Vectors paramsDist) => CheckParamsDist(paramsDist) ? Laplace(paramsDist[0], paramsDist[1], _random) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist) {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Laplace(paramsDist[0], paramsDist[1], _random);
            return result;
        }
        public static double Generate(in Random rand) => Laplace(0, 1, rand);
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParams(paramsDist) ? Laplace(paramsDist[0], paramsDist[1], rand) : null;
        public static Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand) {
            if (!CheckParams(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Laplace(paramsDist[0], paramsDist[1], rand);
            return result;
        }

        public static Moment Mean => (paramDist) => paramDist[0];
        public static Moment Var => (paramDist) => 2 * paramDist[1] * paramDist[1];

        public string[] NameParameters => ["Сдвиг", "Маштаб"];

        public (double?, double?)[]? BoundsParameters => [(null, null), (0, null)];
    }

    [DistributionName("Распределение Коши", TypeDisribution.Cauchy)]
    public class CauchyDistribution(int? seed = null) : IRandomDistribution
    {
        public int CountParametrsDistribution => 2;
        public Vectors DefaultParametrs => new([0, 1]);

        public string Name => "Коши";

        private readonly Random _random = seed is null ? new Random() : new Random((int) seed);
        private static double Cauchy(double loc, double scale, double u) => loc + scale * Math.Tan(double.Pi * (-1.5 + u * 2));
        private static bool CheckParams(Vectors paramsDist) => paramsDist.Size == 2 && paramsDist[1] > 0;
        public bool CheckParamsDist(Vectors paramsDist) => CheckParams(paramsDist);
        public double Generate() => Cauchy(0, 1, _random.NextDouble());
        public double? Generate(Vectors paramsDist) => CheckParamsDist(paramsDist) ? Cauchy(paramsDist[0], paramsDist[1], _random.NextDouble()) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist) 
        {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Cauchy(paramsDist[0], paramsDist[1], _random.NextDouble());
            return result;
        }

        public static double Generate(in Random rand) => Cauchy(0, 1, rand.NextDouble());
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParams(paramsDist) ? Cauchy(paramsDist[0], paramsDist[1], rand.NextDouble()) : null;
        public static Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand) {
            if (!CheckParams(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Cauchy(paramsDist[0], paramsDist[1], rand.NextDouble());
            return result;
        }

        public static Moment Mean => (paramDist) => null;
        public static Moment Var => (paramDist) => null;

        public string[] NameParameters => ["Сдвиг", "Маштаб"];

        public (double?, double?)[]? BoundsParameters => [(null, null), (0, null)];
    }
    
    [DistributionName("Нормальное распределение", TypeDisribution.Normal)]
    public class NormalDistribution(int? seed = null) : IRandomDistribution
    {
        public int CountParametrsDistribution => 2;
        public Vectors DefaultParametrs => new([0, 1]);

        public string Name => "Нормальное";
        public static string? StaticName => "Нормальное";

        private readonly Random _random = seed is null ? new Random() : new Random((int) seed);
        private static double Uniform(double a, double b, in Random rand) => a + (b - a) * rand.NextDouble();
        private static double Normal(double loc, double scale, in Random rand) {
            double e1 = Uniform(-1, 1, rand), e2 = Uniform(-1, 1, rand);
            var s = e1 * e1 + e2 * e2;
            while (s > 1 || s < double.Epsilon)
            {
                e1 = Uniform(-1, 1, rand); e2 = Uniform(-1, 1, rand);
                s = e1 * e1 + e2 * e2;
            }
            return e1 * Math.Sqrt(-2 * Math.Log(s) / s) * scale + loc;
        }
        private static bool CheckParams(Vectors paramsDist) => paramsDist.Size == 2 & paramsDist[1] > 0;
        public bool CheckParamsDist(Vectors paramsDist) => CheckParams(paramsDist);
        public double Generate() => Normal(0, 1, _random);
        public double? Generate(Vectors paramsDist) => CheckParamsDist(paramsDist) ? Normal(paramsDist[0], paramsDist[1], _random) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist) 
        {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            int count_1 = shape.Item1 / 2, count_2 = shape.Item2 / 2;
            double val;
            for (var i = 0; i < shape.Item1; i++)
            {
                for (var j = 0; j < count_2; j++)
                {
                    double e1 = Uniform(-1, 1, _random), e2 = Uniform(-1, 1, _random);
                    var s = e1 * e1 + e2 * e2;
                    while (s > 1 || s < double.Epsilon)
                    {
                        e1 = Uniform(-1, 1, _random); e2 = Uniform(-1, 1, _random);
                        s = e1 * e1 + e2 * e2;
                    }
                    val = Math.Sqrt(-2 * Math.Log(s) / s) * paramsDist[1];
                    result[i, 2 * j] = e1 * val + paramsDist[0];
                    result[i, 2 * j + 1] = e2 * val + paramsDist[0];
                }
            }
            if (shape.Item2 % 2 == 1)
            {
                for (var i = 0; i < count_1; i++)
                {
                    double e1 = Uniform(-1, 1, _random), e2 = Uniform(-1, 1, _random);
                    var s = e1 * e1 + e2 * e2;
                    while (s > 1 || s < double.Epsilon)
                    {
                        e1 = Uniform(-1, 1, _random); e2 = Uniform(-1, 1, _random);
                        s = e1 * e1 + e2 * e2;
                    }
                    val = Math.Sqrt(-2 * Math.Log(s) / s) * paramsDist[1];
                    result[2 * i, -1] = e1 * val + paramsDist[0];
                    result[2 * i + 1, -1] = e2 * val + paramsDist[0];
                }
                if (shape.Item1 % 2 == 1) result[-1, -1] = Normal(paramsDist[0], paramsDist[1], _random);
            }
            return result;
        }

        public static double Generate(in Random rand) => Normal(0, 1, rand);
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParams(paramsDist) ? Normal(paramsDist[0], paramsDist[1], rand) : null;
        public static Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand) {
            if (!CheckParams(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            int count_1 = shape.Item1 / 2, count_2 = shape.Item2 / 2;
            double val;
            for (var i = 0; i < shape.Item1; i++)
            {
                for (var j = 0; j < count_2; j++)
                {
                    double e1 = Uniform(-1, 1, rand), e2 = Uniform(-1, 1, rand);
                    var s = e1 * e1 + e2 * e2;
                    while (s > 1 || s < double.Epsilon)
                    {
                        e1 = Uniform(-1, 1, rand); e2 = Uniform(-1, 1, rand);
                        s = e1 * e1 + e2 * e2;
                    }
                    val = Math.Sqrt(-2 * Math.Log(s) / s) * paramsDist[1];
                    result[i, 2 * j] = e1 * val + paramsDist[0];
                    result[i, 2 * j + 1] = e2 * val + paramsDist[0];
                }
            }
            if (shape.Item2 % 2 == 1)
            {
                for (var i = 0; i < count_1; i++)
                {
                    double e1 = Uniform(-1, 1, rand), e2 = Uniform(-1, 1, rand);
                    var s = e1 * e1 + e2 * e2;
                    while (s > 1 || s < double.Epsilon)
                    {
                        e1 = Uniform(-1, 1, rand); e2 = Uniform(-1, 1, rand);
                        s = e1 * e1 + e2 * e2;
                    }
                    val = Math.Sqrt(-2 * Math.Log(s) / s) * paramsDist[1];
                    result[2 * i, -1] = e1 * val + paramsDist[0];
                    result[2 * i + 1, -1] = e2 * val + paramsDist[0];
                }
                if (shape.Item1 % 2 == 1) result[-1, -1] = Normal(paramsDist[0], paramsDist[1], rand);
            }
            return result;
        }

        public static Moment Mean => (paramDist) => paramDist[0];
        public static Moment Var => (paramDist) => paramDist[1] * paramDist[1];

        public string[] NameParameters => ["Сдвиг", "Маштаб"];

        public (double?, double?)[]? BoundsParameters => [(null, null), (0, null)];
    }
    
    [DistributionName("Гамма распределение", TypeDisribution.Gamma)]
    public class GammaDistribution(int? seed = null) : IRandomDistribution 
    {
        public int CountParametrsDistribution => 3;
        public Vectors DefaultParametrs => new([0, 1, 1]);

        public string Name => "Гамма";
        public static string? StaticName => "Гамма";


        private readonly Random _random = seed is null ? new Random() : new Random((int) seed);
        
#pragma warning disable CS8629 // Тип значения, допускающего NULL, может быть NULL.
        private static double Uniform(Vectors paramDist, in Random rand) => (double) UniformDistribution.Generate(paramDist, rand);
#pragma warning restore CS8629 // Тип значения, допускающего NULL, может быть NULL.
        private static double Exponential(in Random rand) => ExponentialDistribution.Generate(rand);
        private static double Normal(in Random rand) => NormalDistribution.Generate(rand);

        private static double GA1(in Random rand, int k = 1)
        {
            var res = 0.0;
            for (var i = 0; i < k; i++)
                res += Exponential(rand);
            return res;
        }
        private static double GA2(in Random rand ,double k = 0.5)
        {
            var res = Normal(rand);
            var i = 1;
            while (i < k)
            {
                res += Exponential(rand); i++;
            }
            return res;
        }
        private static double GS(in Random rand, double k)
        {
            double res, u, w;
            var iter = 0;
            do
            {
                u = Uniform(new Vectors([0, 1 + k / double.E]), rand);
                w = Exponential(rand);
                if (u <= 1)
                {
                    res = Math.Pow(u, 1.0 / k);
                    if (res <= w)
                        return res;
                }
                else
                {
                    res = -Math.Log((1 - u) / k + 1.0 / double.E);
                    if ((1 - k) * Math.Log(res) <= w)
                        return res;
                }
            } while (++iter < 1e9);
            throw new Exception("Failed to calculate GS");
        }
        private static double GF(in Random rand, double k)
        {
            double e1, e2;
            do
            {
                e1 = Exponential(rand); e2 = Exponential(rand);
            } while (e2 < (k - 1) * (e1 - Math.Log(e1) - 1));
            return k * e1;
        }

        private static double MAT(in Random rand, double k) {
            double d = k - 1.0 / 3;
            double c = 3 * double.Sqrt(d);
            int iter = 0;
            do {
                double n;
                do {
                    n = Normal(rand);
                } while (n <= -c);
                double v = 1 + n / c;
                v = v * v * v;
                n *= n;
                double u = rand.NextDouble();
                if (u < (1.0 - 0.331 * n * n) || (double.Log(u) < (0.5 * n + d * (1.0 - v + double.Log(v)))))
                    return d * v;
            
            } while (++iter <= 1e9);
            throw new Exception("Gamma distribution: sampling failed");
        }

        private static double Gamma(double loc, double scale, double k, in Random rand) {
            double result;
            if (double.IsInteger(k) && k < 5)
                result = GA1(rand, (int) k);
            else if (double.IsInteger(2 * k) && k < 5)
                result = GA2(rand, k);
            else if (k < 1)
                result = GS(rand, k);
            else if (k > 1 && k < 3)
                result = GF(rand, k);
            else
            {
                //var constant = InitConstantGO(k);
                //result = GO(constant, rand);
                result = MAT(rand, k);
            }
            return scale * result + loc;
        }

        private static bool CheckParams(Vectors paramsDist) => paramsDist.Size == 3 && paramsDist[1] > 0 && paramsDist[2] > 0;
        public bool CheckParamsDist(Vectors paramsDist) => CheckParams(paramsDist);
        public double Generate() => GA1(_random);
        public double? Generate(Vectors paramsDist) => CheckParamsDist(paramsDist) ? Gamma(paramsDist[0], paramsDist[1], paramsDist[2], _random) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist) 
        {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Gamma(paramsDist[0], paramsDist[1], paramsDist[2], _random);
            return result;
        }
        public static double Generate(in Random rand) => GA1(rand);
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParams(paramsDist) ? Gamma(paramsDist[0], paramsDist[1], paramsDist[2], rand) : null;
        public static Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand) {
            if (!CheckParams(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Gamma(paramsDist[0], paramsDist[1], paramsDist[2], rand);
            return result;
        }

        public static Moment Mean => (paramDist) => paramDist[0] + paramDist[1] * paramDist[2];
        public static Moment Var => (paramDist) => paramDist[1] * paramDist[1] * paramDist[2];

        public string[] NameParameters => ["Сдвиг","Маштаб", "Форма"];

        public (double?, double?)[]? BoundsParameters => [(null, null), (0, null), (0, null)];
    }

    public static class LinespaceRandom
    {
        public static T[] Shuffle<T>(T[] list, in Random rand)
        {
            var n = list.Length;
            while (n > 1)
            {
                n--;
                var k = rand.Next(n + 1);
                (list[n], list[k]) = (list[k], list[n]);
            }
            return list;
        }

        public static Vectors Generate((int, int) shape, (double, double)[] intervals, in Random rand)
        {
            if (intervals.Length == 1) {
                (double, double)[] intervalsObservations = new (double, double)[shape.Item2];
                for (int i = 0; i < shape.Item2; i++)
                    intervalsObservations[i] = intervals[0];
                intervals = intervalsObservations;
            }
            
            if (intervals.Length != shape.Item2)
                throw new ArgumentException("Количество элементов в Intervals должно совпадать с количеством столбцов");

            var step = new double[shape.Item2];
            for (var i = 0; i < shape.Item2; i++)
                step[i] = (intervals[i].Item2 - intervals[i].Item1) / (shape.Item1 - 1);

            var values = new int[shape.Item1];
            for (var i = 0; i < shape.Item1; i++)
                values[i] = i;

            var result = Vectors.InitVectors(shape);
            int[] copyValues;
            for (var j = 0; j < shape.Item2; j++)
            {
                copyValues = Shuffle((int[]) values.Clone(), rand);
                for (var i = 0; i < shape.Item1; i++)
                    result[i, j] = intervals[j].Item1 + copyValues[i] * step[j];
            }
            return result;
        }
    }

    public static class Linespace
    {

        public static Vectors Generate((int, int) shape, (double, double)[] intervals, in Random rand)
        {
            var x = new Vectors(GenerateGridPoints(intervals, n: shape.Item1, m: shape.Item2));
            x.ShaffleRows(rand);
            return x;
        }

        public static double[][] GenerateGridPoints((double start, double end)[] boundaries, int n, int m)
        {
            // 1. Вычисляем количество точек на ось (k)
            int k = (int) Math.Floor(Math.Pow(n, 1.0 / m));
            int gridPointsCount = (int) Math.Pow(k, m);
            int remainingPoints = n - gridPointsCount;

            // 2. Генерируем равномерную сетку
            List<double[]> points = [];

            var expandedBoundaries = boundaries.Length == 1
                    ? Enumerable.Repeat(boundaries[0], m).ToArray()
                    : boundaries;

            if (k > 1)
            {
                double[][] gridPoints = GenerateUniformGrid(expandedBoundaries, k);
                points.AddRange(gridPoints);
            }
            else
            {
                // Если k=1, добавляем центр (если нужно)
                if (n > 0)
                {
                    double[] centerPoint;
                    if (boundaries.Length == 1) {
                        centerPoint = Enumerable.Repeat((boundaries[0].end + boundaries[0].start) / 2, m).ToArray();
                    }
                    else
                    {
                        centerPoint = new double[m];
                        for (int i = 0; i < m; i++) {
                            centerPoint[i] = (boundaries[i].end + boundaries[i].start) / 2;
                        }
                    }
                    points.Add(centerPoint);
                    remainingPoints = n - 1;
                }
            }

            // 3. Добавляем вершины гиперкуба (с повторением при необходимости)
            if (remainingPoints > 0)
            {
                double[][] vertices = GenerateHypercubeVertices(expandedBoundaries);

                for (int i = 0; i < remainingPoints; i++)
                {
                    // Циклически выбираем вершины
                    points.Add(vertices[i % vertices.Length]);
                }
            }

            return [.. points];
        }

        private static double[][] GenerateUniformGrid((double start, double end)[] boundaries, int k)
        {
            // Генерируем точки для каждого измерения
            double[][] axisPoints = boundaries
                .Select(b => Enumerable.Range(0, k)
                    .Select(i => b.start + i * (b.end - b.start) / (k - 1))
                    .ToArray())
                .ToArray();

            // Общее количество точек (k^m)
            int totalPoints = (int) Math.Pow(k, boundaries.Length);

            // Массив для хранения всех точек
            double[][] gridPoints = new double[totalPoints][];

            // Массив индексов для перебора комбинаций
            int[] indices = new int[boundaries.Length];

            for (int i = 0; i < totalPoints; i++)
            {
                // Создаем новую точку
                double[] point = new double[boundaries.Length];
                for (int dim = 0; dim < boundaries.Length; dim++)
                {
                    point[dim] = axisPoints[dim][indices[dim]];
                }

                gridPoints[i] = point;

                // Обновляем индексы (как в многомерном счетчике)
                for (int dim = boundaries.Length - 1; dim >= 0; dim--)
                {
                    if (++indices[dim] < k)
                    {
                        break;
                    }
                    indices[dim] = 0;
                }
            }

            return gridPoints;
        }

        private static double[][] GenerateHypercubeVertices((double min, double max)[] boundaries)
        {
            int m = boundaries.Length;
            int vertexCount = (int) Math.Pow(2, m);
            double[][] vertices = new double[vertexCount][];

            for (int i = 0; i < vertexCount; i++)
            {
                double[] vertex = new double[m];
                for (int dim = 0; dim < m; dim++)
                {
                    // Используем биты числа i для выбора min или max в каждом измерении
                    vertex[dim] = ((i >> dim) & 1) == 0 ? boundaries[dim].min : boundaries[dim].max;
                }
                vertices[i] = vertex;
            }

            return vertices;
        }
    }
}