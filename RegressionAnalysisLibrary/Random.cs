using System.Reflection;

namespace RegressionAnalysisLibrary
{
    public class DistributionFactory
    {
        private readonly Dictionary<TypeDisribution, IRandomDistribution> _distributions;

        public DistributionFactory(int? seed = null)
        {
            _distributions = [];
            // Загружаем все генераторы, помеченные атрибутом DistributionNameAttribute
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
                    // Создаём экземпляр генератора и регистрируем его по enum-типу
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
    // Типы поддерживаемых распределений
    public enum TypeDisribution
    {
        Uniform,
        Normal,
        Exponential,
        Laplace,
        Cauchy,
        Gamma
    }
    // Атрибут для регистрации распределения по имени и типу
    [AttributeUsage(AttributeTargets.Class, Inherited = false, AllowMultiple = false)]
    public class DistributionNameAttribute(string name, TypeDisribution type) : Attribute
    {
        public string Name { get; } = name;
        public TypeDisribution Type { get; } = type;
    }
    // Общий интерфейс для всех генераторов случайных величин
    public interface IRandomDistribution
    {
        public string Name { get; }
        public Random Generator { get; set; }
        public int CountParametrsDistribution { get; }
        public string[] NameParameters { get; }
        public (double?, double?)[]? BoundsParameters { get; }
        Vectors DefaultParametrs { get; }
        public double Generate(); // Без параметров
        public double? Generate(Vectors paramsDist); // С параметрами
        public Vectors? Generate((int, int) shape, Vectors paramsDist); // Матрица значений
        public Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand); // Матрица с кастомным генератором
        public bool CheckParamsDist(Vectors paramsDist); // Проверка передаваемых параметров
    }
    // Равномерное распределение
    [DistributionName("Равномерное распределение", TypeDisribution.Uniform)]
    public class UniformDistribution(int? seed = null) : IRandomDistribution
    {
        public int CountParametrsDistribution => 2;
        public Vectors DefaultParametrs => new([0, 1]);

        public string Name => "Uniform";

        public Random Generator { get; set; } = seed is null ? new Random() : new Random((int) seed);

        private static bool CheckParams(Vectors paramsDist)
        {
            return paramsDist.Size == 2 && paramsDist[0] < paramsDist[1];
        }
        // Проверка корректности параметров (a < b)
        public bool CheckParamsDist(Vectors paramsDist) => CheckParams(paramsDist);
        // Генерация случайного значения в интервале [a, b]
        private static double Uniform(double a, double b, in Random rand) => a + rand.NextDouble() * (b - a);
        public static double Generate(in Random rand) => rand.NextDouble();
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParams(paramsDist) ? Uniform(paramsDist[0], paramsDist[1], rand) : null;
        public double Generate() => Generator.NextDouble();
        public double? Generate(Vectors paramsDist) => CheckParamsDist(paramsDist) ? UniformDistribution.Uniform(paramsDist[0], paramsDist[1], Generator) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist)
        {
            if (!CheckParamsDist(paramsDist)) return null;
            var vec = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    vec[i, j] = Uniform(paramsDist[0], paramsDist[1], Generator);
            return vec;
        }

        public Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand)
        {
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
    public class ExponentialDistribution(int? seed = null) : IRandomDistribution
    {

        public int CountParametrsDistribution => 2;
        public Vectors DefaultParametrs => new([0, 1]);

        public string Name => "Exponential";
        public Random Generator { get; set; } = seed is null ? new Random() : new Random((int) seed);
        // Генерация случайной экспоненциальной величины
        private static double Exponential(double loc, double scale, double u) => -Math.Log(1 - u) * scale + loc;
        private static bool CheckParams(Vectors paramsDist) => paramsDist.Size == 2 && paramsDist[1] > 0;
        // Проверка корректности параметров (beta > 0)
        public bool CheckParamsDist(Vectors paramsDist) => CheckParams(paramsDist);
        public double Generate() => Exponential(0, 1, Generator.NextDouble());
        public double? Generate(Vectors paramsDist) => CheckParamsDist(paramsDist) ? Exponential(paramsDist[0], paramsDist[1], Generator.NextDouble()) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist)
        {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Exponential(paramsDist[0], paramsDist[1], Generator.NextDouble());
            return result;
        }
        public static double Generate(in Random rand) => Exponential(0, 1, rand.NextDouble());
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParams(paramsDist) ? Exponential(paramsDist[0], paramsDist[1], rand.NextDouble()) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand)
        {
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

        public string Name => "Laplace";

        public Random Generator { get; set; } = seed is null ? new Random() : new Random((int) seed);
        // Генерация случайной величины, подчиняющиеся закону Лапласа
        private static double Laplace(double loc, double scale, in Random rand)
        {
            return loc - scale * double.Sign(-1 + rand.NextDouble() * 2) * double.Log(1 - rand.NextDouble());
        }
        private static bool CheckParams(Vectors paramsDist) => paramsDist.Size == 2 && paramsDist[1] > 0;
        // Проверка корректности параметров (beta > 0)
        public bool CheckParamsDist(Vectors paramsDist) => CheckParams(paramsDist);
        public double Generate() => Laplace(0, 1, Generator);
        public double? Generate(Vectors paramsDist) => CheckParamsDist(paramsDist) ? Laplace(paramsDist[0], paramsDist[1], Generator) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist)
        {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Laplace(paramsDist[0], paramsDist[1], Generator);
            return result;
        }
        public static double Generate(in Random rand) => Laplace(0, 1, rand);
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParams(paramsDist) ? Laplace(paramsDist[0], paramsDist[1], rand) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand)
        {
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

        public string Name => "Cauchy";

        public Random Generator { get; set; } = seed is null ? new Random() : new Random((int) seed);
        // Генерация случайной величины, подчиняющиеся закону Коши
        private static double Cauchy(double loc, double scale, double u) => loc + scale * Math.Tan(double.Pi * (-1.5 + u * 2));
        private static bool CheckParams(Vectors paramsDist) => paramsDist.Size == 2 && paramsDist[1] > 0;
        // Проверка корректности параметров (gamma > 0)
        public bool CheckParamsDist(Vectors paramsDist) => CheckParams(paramsDist);
        public double Generate() => Cauchy(0, 1, Generator.NextDouble());
        public double? Generate(Vectors paramsDist) => CheckParamsDist(paramsDist) ? Cauchy(paramsDist[0], paramsDist[1], Generator.NextDouble()) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist)
        {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Cauchy(paramsDist[0], paramsDist[1], Generator.NextDouble());
            return result;
        }

        public static double Generate(in Random rand) => Cauchy(0, 1, rand.NextDouble());
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParams(paramsDist) ? Cauchy(paramsDist[0], paramsDist[1], rand.NextDouble()) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand)
        {
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
    // Нормальное распределение с использованием метода Бокса-Мюллера
    [DistributionName("Нормальное распределение", TypeDisribution.Normal)]
    public class NormalDistribution(int? seed = null) : IRandomDistribution
    {
        public int CountParametrsDistribution => 2;
        public Vectors DefaultParametrs => new([0, 1]);

        public string Name => "Normal";

        public Random Generator { get; set; } = seed is null ? new Random() : new Random((int) seed);
        private static double Uniform(double a, double b, in Random rand) => a + (b - a) * rand.NextDouble();
        // Генерация одного значения методом Бокса-Мюллера
        private static double Normal(double loc, double scale, in Random rand)
        {
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
        // Проверка корректности параметров (sigma > 0)
        public bool CheckParamsDist(Vectors paramsDist) => CheckParams(paramsDist);
        public double Generate() => Normal(0, 1, Generator);
        public double? Generate(Vectors paramsDist) => CheckParamsDist(paramsDist) ? Normal(paramsDist[0], paramsDist[1], Generator) : null;
        // При генерации матрицы нормальных значений используем парную генерацию для ускорения
        public Vectors? Generate((int, int) shape, Vectors paramsDist)
        {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            int count_1 = shape.Item1 / 2, count_2 = shape.Item2 / 2;
            double val;
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < count_2; j++)
                {
                    double e1 = Uniform(-1, 1, Generator), e2 = Uniform(-1, 1, Generator);
                    var s = e1 * e1 + e2 * e2;
                    while (s > 1 || s < double.Epsilon)
                    {
                        e1 = Uniform(-1, 1, Generator); e2 = Uniform(-1, 1, Generator);
                        s = e1 * e1 + e2 * e2;
                    }
                    val = Math.Sqrt(-2 * Math.Log(s) / s) * paramsDist[1];
                    result[i, 2 * j] = e1 * val + paramsDist[0];
                    result[i, 2 * j + 1] = e2 * val + paramsDist[0];
                }
            if (shape.Item2 % 2 == 1)
            {
                for (var i = 0; i < count_1; i++)
                {
                    double e1 = Uniform(-1, 1, Generator), e2 = Uniform(-1, 1, Generator);
                    var s = e1 * e1 + e2 * e2;
                    while (s > 1 || s < double.Epsilon)
                    {
                        e1 = Uniform(-1, 1, Generator); e2 = Uniform(-1, 1, Generator);
                        s = e1 * e1 + e2 * e2;
                    }
                    val = Math.Sqrt(-2 * Math.Log(s) / s) * paramsDist[1];
                    result[2 * i, -1] = e1 * val + paramsDist[0];
                    result[2 * i + 1, -1] = e2 * val + paramsDist[0];
                }
                if (shape.Item1 % 2 == 1) result[-1, -1] = Normal(paramsDist[0], paramsDist[1], Generator);
            }
            return result;
        }

        public static double Generate(in Random rand) => Normal(0, 1, rand);
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParams(paramsDist) ? Normal(paramsDist[0], paramsDist[1], rand) : null;
        // При генерации матрицы нормальных значений используем парную генерацию для ускорения
        public Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand)
        {
            if (!CheckParams(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            int count_1 = shape.Item1 / 2, count_2 = shape.Item2 / 2;
            double val;
            for (var i = 0; i < shape.Item1; i++)
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

        public string Name => "Gamma";

        public Random Generator { get; set; } = seed is null ? new Random() : new Random((int) seed);

#pragma warning disable CS8629 // Тип значения, допускающего NULL, может быть NULL.
        private static double Uniform(Vectors paramDist, in Random rand) => (double) UniformDistribution.Generate(paramDist, rand);
#pragma warning restore CS8629 // Тип значения, допускающего NULL, может быть NULL.
        private static double Exponential(in Random rand) => ExponentialDistribution.Generate(rand);
        private static double Normal(in Random rand) => NormalDistribution.Generate(rand);

        //GA1 - сумма стандартных экспоненциальных величин
        private static double GA1(in Random rand, int k = 1)
        {
            var res = 0.0;
            for (var i = 0; i < k; i++)
                res += Exponential(rand);
            return res;
        }
        //GA2 - GA1 + половина квадрана нормальной величины
        private static double GA2(in Random rand, double k = 0.5)
        {
            var res = Normal(rand);
            var i = 1;
            while (i < k)
            {
                res += Exponential(rand); i++;
            }
            return res;
        }

        //Ahrens and Dieter, 1974
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

        //Marsaglia and Tsang’s Method 
        private static double MAT(in Random rand, double k)
        {
            var d = k - 1.0 / 3;
            var c = 3 * double.Sqrt(d);
            var iter = 0;
            do
            {
                double n;
                do
                    n = Normal(rand);
                while (n <= -c);
                var v = 1 + n / c;
                v = v * v * v;
                n *= n;
                var u = rand.NextDouble();
                if (u < 1.0 - 0.331 * n * n || double.Log(u) < 0.5 * n + d * (1.0 - v + double.Log(v)))
                    return d * v;

            } while (++iter <= 1e9);
            throw new Exception("Gamma distribution: sampling failed");
        }
        // Используются методы: GA1 (сумма экспоненциальных), GS, MAT и др.
        private static double Gamma(double loc, double scale, double k, in Random rand)
        {
            var result = double.IsInteger(k) && k < 5
                ? GA1(rand, (int) k)
                : double.IsInteger(2 * k) && k < 5 ? GA2(rand, k) : k < 1 ? GS(rand, k) : MAT(rand, k);
            return scale * result + loc;
        }

        private static bool CheckParams(Vectors paramsDist) => paramsDist.Size == 3 && paramsDist[1] > 0 && paramsDist[2] > 0;
        // Проверка корректности параметров (theta > 0 и k > 0)
        public bool CheckParamsDist(Vectors paramsDist) => CheckParams(paramsDist);
        public double Generate() => GA1(Generator);
        public double? Generate(Vectors paramsDist) => CheckParamsDist(paramsDist) ? Gamma(paramsDist[0], paramsDist[1], paramsDist[2], Generator) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist)
        {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Gamma(paramsDist[0], paramsDist[1], paramsDist[2], Generator);
            return result;
        }
        public static double Generate(in Random rand) => GA1(rand);
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParams(paramsDist) ? Gamma(paramsDist[0], paramsDist[1], paramsDist[2], rand) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand)
        {
            if (!CheckParams(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Gamma(paramsDist[0], paramsDist[1], paramsDist[2], rand);
            return result;
        }

        public static Moment Mean => (paramDist) => paramDist[0] + paramDist[1] * paramDist[2];
        public static Moment Var => (paramDist) => paramDist[1] * paramDist[1] * paramDist[2];

        public string[] NameParameters => ["Сдвиг", "Маштаб", "Форма"];

        public (double?, double?)[]? BoundsParameters => [(null, null), (0, null), (0, null)];
    }

    public static class LinespaceRandom
    {
        // Перемешивает массив значений (Fisher–Yates shuffle)
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
        // Генерация таблицы с равномерным покрытием интервалов (размещение наблюдений в линиях)
        public static Vectors Generate((int, int) shape, (double, double)[] intervals, in Random rand)
        {
            if (intervals.Length == 1)
            {
                var intervalsObservations = new (double, double)[shape.Item2];
                for (var i = 0; i < shape.Item2; i++)
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
}