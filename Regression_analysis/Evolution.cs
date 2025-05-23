using System.Reflection;

using Regression_analysis;

namespace RegressionAnalysisLibrary
{
    public interface IParameterEstimator
    {
        string Name { get; }
        public abstract Vectors EstimateParameters(IModel model, Vectors[] otherParameters);
    }


    [AttributeUsage(AttributeTargets.Class, Inherited = false, AllowMultiple = false)]
    public class EvolutionAttribute : Attribute
    {
        public string Name { get; }

        public EvolutionAttribute(string name)
        {
            Name = name;
        }
    }


    public class EvolutionFactory
    {
        private readonly Dictionary<string, IParameterEstimator> _evolutions;

        public EvolutionFactory()
        {
            _evolutions = [];
            LoadEvolutions();
        }

        private void LoadEvolutions()
        {
            var assembly = Assembly.GetExecutingAssembly();

            foreach (var type in assembly.GetTypes()
                .Where(t => typeof(IParameterEstimator).IsAssignableFrom(t)
                        && !t.IsInterface
                        && !t.IsAbstract))
            {
                var attr = type.GetCustomAttribute<EvolutionAttribute>();
                if (attr == null) continue;

                try
                {
                    var instance = Activator.CreateInstance(type) as IParameterEstimator;
#pragma warning disable CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
                    _evolutions[attr.Name] = instance;
#pragma warning restore CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to create {attr.Name}: {ex.Message}");
                }
            }
        }
        public IParameterEstimator GetEvolution(string type)
        {
            return _evolutions.TryGetValue(type, out var evolution)
                ? evolution
                : throw new KeyNotFoundException($"Evolution {type} not found");
        }

        public IEnumerable<(string Name, IParameterEstimator Instance)> GetAllEvolutions()
        {
            return _evolutions.Select(kv => (kv.Key, kv.Value));
        }
    }

    [Evolution("Метод наименьших квадратов (МНК)")]
    public class MNKEstimator : IParameterEstimator
    {
        public string Name => "MNK";

        public Vectors EstimateParameters(IModel model, Vectors[] otherParameters)
        {
            var x = otherParameters[1];
            var y = otherParameters[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Вектор 'x' не соответствует количеству факторов в моделе 'model'");

            if (!y.IsVector() || y.Size != x.Shape.Item1)
                throw new ArgumentException("Вектор 'y' имеет не правильный вид.");

            var matrixX = model.CreateMatrixX(x);

            Vectors estimatedTheta;
            try
            {
                var tMatrixX = matrixX.T();
                estimatedTheta = (Vectors.Inv(tMatrixX & matrixX) & tMatrixX & y.T()).T();
            }
            catch
            {
                throw new Exception("Не правильные данные");
            }
            return estimatedTheta;
        }
    }

    [Evolution("Метод максимального правдоподобия (ММП)")]
    public class MMPEstimator : IParameterEstimator
    {
        public MMPConfiguration Config { get; set; } = MMPConfigLoader.Normal();

        public string Name => "MMP";

        public Vectors EstimateParameters(IModel model, Vectors[] otherParameters)
        {
            if (Config.Functions.Name == "Normal")
            {
                var x = otherParameters[1];
                var y = otherParameters[2];

                if (x.Shape.Item2 != model.CountFacts)
                    throw new ArgumentException("Вектор 'x' не соответствует количеству факторов в моделе 'model'");

                if (!y.IsVector() || y.Size != x.Shape.Item1)
                    throw new ArgumentException("Вектор 'y' имеет не правильный вид.");

                var matrixX = model.CreateMatrixX(x);

                Vectors estimatedTheta;
                try
                {
                    var tMatrixX = matrixX.T();
                    estimatedTheta = Vectors.Inv(tMatrixX & matrixX) & tMatrixX & y.T();
                }
                catch
                {
                    throw new Exception("Не правильные данные");
                }
                return estimatedTheta;
            }
            else
                if (Config.MNKEstuminate)
            {
                Vectors initParams;
                try
                {
                    var matrixX = model.CreateMatrixX(otherParameters[1]);
                    var tMatrixX = matrixX.T();
                    var mean = Config.Mean(otherParameters[0]);
                    initParams = mean is null
                        ? Vectors.Inv(tMatrixX & matrixX) & tMatrixX & otherParameters[2].T()
                        : Vectors.Inv(tMatrixX & matrixX) & tMatrixX & (otherParameters[2] - (double) mean).T();
                }
                catch
                {
                    throw new Exception("Не правильные данные");
                }

                return Config.IsMultiIterationOptimisation
                    ? Config.Oprimizator.OptimisateRandomInit(
                        Config.Functions.LogLikelihood,
                        Config.Functions.Gradient,
                        model,
                        otherParameters,
                        Config.Tolerance,
                        maxIter: Config.MaxIteration,
                        seed: Config.Seed,
                        x0: initParams.T()
                        ).MinPoint
                    : Config.Oprimizator.Optimisate(
                        Config.Functions.LogLikelihood,
                        Config.Functions.Gradient,
                        model,
                        initParams.T(),
                        otherParameters,
                        Config.Tolerance,
                        maxIter: Config.MaxIteration
                        ).MinPoint
                    ;
            }
            else
                return Config.IsMultiIterationOptimisation
                    ? Config.Oprimizator.OptimisateRandomInit(
                        Config.Functions.LogLikelihood,
                        Config.Functions.Gradient,
                        model,
                        otherParameters,
                        Config.Tolerance,
                        maxIter: Config.MaxIteration,
                        seed: Config.Seed
                        ).MinPoint
                    : Config.Oprimizator.Optimisate(
                        Config.Functions.LogLikelihood,
                        Config.Functions.Gradient,
                        model,
                        Vectors.Zeros((1, model.CountRegressor)),
                        otherParameters,
                        Config.Tolerance,
                        maxIter: Config.MaxIteration
                        ).MinPoint;
        }
    }

    public static class MMPConfigLoader
    {
        public static MMPConfiguration Normal
            (
            bool ismultiiteration = true,
            int maxattepts = 100,
            int maxiteration = 1000,
            double tol = 1e-7,
            int? seed = null
            )
        {
            return new MMPConfiguration()
            {
                Functions = new NormalMMPDistribution(),
                Oprimizator = new DFPOptimizator(),
                Mean = NormalDistribution.Mean,
                IsMultiIterationOptimisation = ismultiiteration,
                MaxAttempts = maxattepts,
                Tolerance = tol,
                MaxIteration = maxiteration,
                Seed = seed
            };
        }
        public static MMPConfiguration Exponential
            (
            bool ismultiiteration = true,
            int maxattepts = 100,
            int maxiteration = 1000,
            double tol = 1e-7,
            int? seed = null
            )
        {
            return new MMPConfiguration()
            {
                Functions = new ExponentialMMPDistribution(),
                Oprimizator = new NelderMeadOptimizator(),
                Mean = ExponentialDistribution.Mean,
                IsMultiIterationOptimisation = ismultiiteration,
                MaxAttempts = maxattepts,
                Tolerance = tol,
                MaxIteration = maxiteration,
                Seed = seed,
            };
        }

        public static MMPConfiguration Laplace
            (
            bool ismultiiteration = true,
            int maxattepts = 100,
            int maxiteration = 1000,
            double tol = 1e-7,
            int? seed = null
            )
        {
            return new MMPConfiguration()
            {
                Functions = new LaplaceMMPDistribution(),
                Oprimizator = new NelderMeadOptimizator(),
                Mean = LaplaceDistribution.Mean,
                IsMultiIterationOptimisation = ismultiiteration,
                MaxAttempts = maxattepts,
                Tolerance = tol,
                MaxIteration = maxiteration,
                Seed = seed
            };
        }
        public static MMPConfiguration Cauchy
            (
                bool ismultiiteration = true,
                int maxattepts = 100,
                int maxiteration = 1000,
                double tol = 1e-7,
                int? seed = null
            )
        {
            return new MMPConfiguration()
            {
                Functions = new CauchyMMPDistribution(),
                Oprimizator = new DFPOptimizator(),
                Mean = CauchyDistribution.Mean,
                IsMultiIterationOptimisation = ismultiiteration,
                MaxAttempts = maxattepts,
                Tolerance = tol,
                MaxIteration = maxiteration,
                Seed = seed,
                MNKEstuminate = false
            };
        }

        public static MMPConfiguration Gamma
            (
                bool ismultiiteration = true,
                int maxattepts = 100,
                int maxiteration = 1000,
                double tol = 1e-7,
                int? seed = null
            )
        {
            return new MMPConfiguration()
            {
                Functions = new GammaMMPDistribution(),
                Oprimizator = new NelderMeadOptimizator(),
                Mean = GammaDistribution.Mean,
                IsMultiIterationOptimisation = ismultiiteration,
                MaxAttempts = maxattepts,
                Tolerance = tol,
                MaxIteration = maxiteration,
                Seed = seed
            };
        }
        public static MMPConfiguration Uniform
            (
                bool ismultiiteration = true,
                int maxattepts = 100,
                int maxiteration = 1000,
                double tol = 1e-7,
                int? seed = null
            )
        {
            return new MMPConfiguration()
            {
                Functions = new UniformMMPDistribution(),
                Oprimizator = new NelderMeadOptimizator(),
                Mean = UniformDistribution.Mean,
                IsMultiIterationOptimisation = ismultiiteration,
                MaxAttempts = maxattepts,
                Tolerance = tol,
                MaxIteration = maxiteration,
                Seed = seed
            };
        }
        public static MMPConfiguration Load(
            TypeDisribution typeDisribution,
            bool ismultiiteration = true,
            int maxattepts = 100,
            int maxiteration = 5000,
            double tol = 1e-7,
            int? seed = null
            )
        {
            switch (typeDisribution)
            {
                case TypeDisribution.Normal:
                    return Normal(ismultiiteration, maxattepts, maxiteration, tol, seed);
                case TypeDisribution.Laplace:
                    return Laplace(ismultiiteration, maxattepts, maxiteration, tol, seed);
                case TypeDisribution.Exponential:
                    return Exponential(ismultiiteration, maxattepts, maxiteration, tol, seed);
                case TypeDisribution.Cauchy:
                    return Cauchy(ismultiiteration, maxattepts, maxiteration, tol, seed);
                case TypeDisribution.Uniform:
                    return Uniform(ismultiiteration, maxattepts, maxiteration, tol, seed);
                case TypeDisribution.Gamma:
                    return Gamma(ismultiiteration, maxattepts, maxiteration, tol, seed);
                default:
                    throw new ArgumentException("Нет такого распределения");
            }
        }
    }

    public static class NumericGradient
    {
        public static Vectors ComputeGradient(LogLikelihoodFunction f, Vectors x, IModel model, Vectors[] parameters, double h = 1e-5)
        {
            var n = x.Size;
            var gradient = Vectors.InitVectors((1, n));
            var xPh = x.Clone();
            var xMh = x.Clone();

            for (var i = 0; i < n; i++)
            {
                // Сохраняем исходное значение
                var originalValue = x[i];

                // Вычисляем f(x + h)
                xPh[i] = originalValue + h;
                var fPh = f(model, xPh, parameters);

                // Вычисляем f(x - h)
                xMh[i] = originalValue - h;
                var fMh = f(model, xMh, parameters);

                // Центральная разностная производная
                gradient[i] = (fPh - fMh) / (2 * h);

                // Восстанавливаем значение
                xPh[i] = originalValue;
                xMh[i] = originalValue;
            }

            return gradient;
        }
    }


    public static class Test
    {
        public static void Main(string[] args)
        {

            var h = "H0";
            var n = 1000;
            var thetaH0 = new Vectors([5, 2, 3, 7]);
            var valueH1 = 1.0 / (n * double.Log(n) * double.Log(n));
            var thetaH1 = new Vectors([5, valueH1, valueH1, valueH1]);
            var paramDistribution = new Vectors([0, 30, 0.7]);
            var numparam = 1;

            var e_teml = GammaDistribution.Generate((1, 100000), paramDistribution, new Random());
            e_teml.SaveToDAT($"D:/Program/Budancev/ОР/Samples/Gamma07.dat", "Gamma (0, 30, 0.7)");
            /*
            
            Vectors planX = new Vectors([[-1, -1, -1],
                                        [1, -1, -1],
                                        [-1, 1, -1],
                                        [1, 1, -1],
                                        [-1, -1, 1],
                                        [1, -1, 1],
                                        [-1, 1, 1],
                                        [1, 1, 1]]);
            Vectors planP = Vectors.Ones((1, 8)) / 8;
            planX = planX * 1e+3;

            var model = new LiniarModel
                        (
                            3,
                            [],
                            [],
                            thetaH0,
                            true
                        );

            var rand = new Random();
            var error = new GammaDistribution();
            var e = error.Generate((1, n), paramDistribution);
            var X = LinespaceRandom.Generate((n, model.CountFacts), [(-1, 1)], rand);
            var y = (model.CreateMatrixX(X) & model.TrueTheta.T()).T() + e;
            var func = new GammaMMKDistribution();

            //var x0 = new Vectors([1, 2, 5, 9]);

            //var opt3 = new CGOptimizator();
            var opt1 = new NelderMeadOptimizator();
            //var opt2 = new DFPOptimizator();

            var MNK = new MNKEstimator();
            var calc_theta = MNK.EstimateParameters(model, [paramDistribution, X, y]);

            var clock = new Stopwatch();
            clock.Start();
            var result0 = opt1.OptimisateRandomInit(
                    func.LogLikelihood,
                    func.Gradient,
                    model,
                    [paramDistribution, X, y],
                    1e-7,
                    x0: calc_theta,
                    maxIter: 1000
                );
            
            clock.Stop();
            Console.WriteLine($"Первое время: {clock.ElapsedMilliseconds}");
            var result1 = opt1.Optimisate(
                func.LogLikelihood,
                    func.Gradient,
                    model,
                    calc_theta,
                    [paramDistribution, X, y],
                    1e-7,
                    maxIter: 1000
                );
            */
            /*
            clock.Restart();
            var result1 = opt2.OptimisateRandomInit(
                    func.LogLikelihood,
                    func.Gradient,
                    model,
                    [paramDistribution, X, y],
                    1e-7,
                    x0: calc_theta
                );
            clock.Stop();
            Console.WriteLine($"Второе время: {clock.ElapsedMilliseconds}");
            
            clock.Restart();
            var result2 = opt3.Optimisate(
                    func.LogLikelihood,
                    func.Gradient,
                    model,
                    calc_theta,
                    [paramDistribution, X, y],
                    1e-7
                );
            clock.Stop();
            Console.WriteLine($"Третье время: {clock.ElapsedMilliseconds}");
            */
            //Console.WriteLine(result0 + "\n");
            //Console.WriteLine(result1 + "\n");
            //Console.WriteLine(result2 + "\n");
            //Console.WriteLine($"Значение функции для первого оптимизатора: {func.LogLikelihood(model, result0.MinPoint, [paramDistribution, X, y])}");
            //Console.WriteLine($"Значение функции для второго оптимизатора: {func.LogLikelihood(model, result1.MinPoint, [paramDistribution, X, y])}");
            //Console.WriteLine($"Значение функции для второго оптимизатора: {func.LogLikelihood(model, result2.MinPoint, [paramDistribution, X, y])}");

            /*
            var rand = new Random();
            var opt1 = new CGOptimizator();
            var opt2 = new NelderMeadOptimizator();
            
            var func = new CauchyMMKDistribution();
            var X = LinespaceRandom.Generate((n, model.CountFacts), [(-10, 10)], rand);
            //var X = RegressionEvaluator.GenerateXFromPlan(planX, planP, n, rand);
            var error = new CauchyDistribution();
            var e = error.Generate((1, n), paramDistribution);

            var y = (model.CreateMatrixX(X) & model.TrueTheta.T()).T() + e;

            //var res = ComparisonMethods.Compare(
            //        model,
            //        new MNKEstimator(),
            //        new MMKEstimator(MMKConfigLoader.Laplace(ismultiiteration: false, maxiteration: 5000)),
            //        [paramDistribution, X, y]

            //    );
            var MNK = new MNKEstimator();
            var calc_theta = MNK.EstimateParameters(model, [paramDistribution, X, y]);
            var clock = new Stopwatch();
            clock.Start();

            var res = opt1.Optimisate(
                func.LogLikelihood,
                func.Gradient,
                model,
                calc_theta,
                //Vectors.Inv(func.Gessian(model, calc_theta, [paramDistribution, X, y])),
                [paramDistribution, X, y],
                eps: 1e-7
            );
            clock.Stop();

            Console.WriteLine((res, clock.ElapsedMilliseconds));

            Console.WriteLine(calc_theta);
            clock.Restart();
            res = opt2.Optimisate(
                func.LogLikelihood,
                func.Gradient,
                model,
                calc_theta,
                [paramDistribution, X, y],
                1e-7
            );
            clock.Stop();
            Console.WriteLine((res, clock.ElapsedMilliseconds));
            */
            //string json = JsonSerializer.Serialize(res, new JsonSerializerOptions { WriteIndented = true, IncludeFields = true});
            //File.WriteAllText($"/home/zodiac/Program/ОР/Samples/CompareMethods_Laplace_{n}.json", json);
            //Console.WriteLine(json);

            /*
            Vectors planX = new Vectors([[-1, -1, -1],
                                        [1, -1, -1],
                                        [-1, 1, -1],
                                        [1, 1, -1],
                                        [-1, -1, 1],
                                        [1, -1, 1],
                                        [-1, 1, 1],
                                        [1, 1, 1]]);
            Vectors planP = Vectors.Ones((1, 8)) / 8;
            planX *= 1e+4;
            int seed = 8745;
            if (h == "H0")
            {
                var clock = new Stopwatch();
                clock.Start();
                var statistic = RegressionEvaluator.FitParameters
                    (
                        model: new LiniarModel
                        (
                            3,
                            [],
                            thetaH0,
                            true
                        ),
                        evolution: new MMKEstimator(MMKConfigLoader.Exponential(ismultiiteration: true, maxiteration: 2000)),
                        //evolution: new MNKEstimator(),
                        countIteration: 2000,
                        countObservations: n,
                        numberParametr: numparam,
                        errorDist: new ExponentialDistribution(),
                        paramsDist: paramDistribution,
                        debug: true,
                        parallel: true,
                        isRound: false,
                        planX: planX,
                        planP: planP,
                        roundDecimals: 5
                    //seed: seed
                    );
                clock.Stop();
                Console.WriteLine();
                Console.WriteLine(clock.ElapsedMilliseconds);
                Console.WriteLine("Готово!");
                statistic.Statistics.SaveToDAT(FormattableString.Invariant($"D:/Program/Budancev/ОР/Samples/H0_Parameters_{numparam}_{1e4}_MMPExponential{n}.dat"), title: "H0 " + statistic.ToString());
                //statistic.Statistics.SaveToDAT(FormattableString.Invariant($"/home/zodiac/Program/ОР/Samples/H0_MMKLaplace{n}.dat"), title: "H0 " + statistic.ToString());
            }
            else if (h == "H1") 
            {
                var clock = new Stopwatch();
                clock.Start();
                var statistic = RegressionEvaluator.Fit
                    (
                        model: new LiniarModel
                        (
                            3,
                            [],
                            thetaH1,
                            true
                        ),
                        evolution: new MMKEstimator(MMKConfigLoader.Laplace()),
                        countIteration: 2000,
                        countObservations: n,
                        
                        errorDist: new LaplaceDistribution(),
                        paramsDist: paramDistribution,
                        debug: true,
                        parallel: true,
                        seed: seed
                    );
                clock.Stop();
                Console.WriteLine();
                Console.WriteLine(clock.ElapsedMilliseconds);
                Console.WriteLine("Готово!");
                statistic.Statistics.SaveToDAT(FormattableString.Invariant($"D:\\Program\\Budancev\\ОР\\Samples\\H1_MMKLaplace{n}_lr.dat"), title: "H1 " + statistic.ToString());
                
            }
            */
        }
    }
}
