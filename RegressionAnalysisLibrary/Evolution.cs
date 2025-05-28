using System.Reflection;

namespace RegressionAnalysisLibrary
{
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
    public interface IParameterEstimator
    {
        string Name { get; }
        public abstract Vectors EstimateParameters(IModel model, Vectors[] otherParameters);
    }


    [AttributeUsage(AttributeTargets.Class, Inherited = false, AllowMultiple = false)]
    public class EvolutionAttribute(string name) : Attribute
    {
        public string Name { get; } = name;
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
                    estimatedTheta = (Vectors.Inv(tMatrixX & matrixX) & tMatrixX & y.T()).T();
                }
                catch
                {
                    throw new Exception("Не правильные данные");
                }
                return estimatedTheta;
            }
            else if (Config.MNKEstuminate)
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
                        ).MinPoint;
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
                IsMultiIterationOptimisation = true,
                MaxAttempts = maxattepts,
                Tolerance = tol,
                MaxIteration = 1000,
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
}