using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Regression_analysis
{
    public interface IParameterEstimator 
    { 
        string Name { get; }
        public abstract Vectors EstimateParameters(IModel model, Vectors[] otherParameters);
    }

    public class MNKEstimator : IParameterEstimator {
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
                estimatedTheta = ((Vectors.Inv(tMatrixX & matrixX) & tMatrixX) & y.T()).T();
            }
            catch
            {
                throw new Exception("Не правильные данные");
            }
            return estimatedTheta;
        }
    }

    public class MMKEstimator(MMKConfiguration config) : IParameterEstimator 
    {
        public readonly MMKConfiguration Config = config ?? throw new ArgumentNullException(nameof(config));

        public string Name => "MMK";

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
                    estimatedTheta = (Vectors.Inv(tMatrixX & matrixX) & tMatrixX) & y.T();
                }
                catch
                {
                    throw new Exception("Не правильные данные");
                }
                return estimatedTheta;
            }
            else 
            {
                return Config.IsMultiIterationOptimisation
                    ? Config.Oprimizator.OptimisateRandomInit(
                        Config.Functions.LogLikelihood,
                        Config.Functions.Gradient,
                        model,
                        otherParameters,
                        (1, model.CountRegressor),
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
                        Config.Tolerance
                        ).MinPoint
                    ;


            }
        }
    }

    public static class MMKConfigLoader 
    {
        public static MMKConfiguration Normal
            (
            bool ismultiiteration = true, 
            int maxattepts = 100, 
            int maxiteration = 1000, 
            double tol = 1e-7, 
            int? seed = null
            ) 
        {
            return new MMKConfiguration()
            {
                Functions = new NormalMMKDistribution(),
                Oprimizator = new DFPOptimizator(),
                TypeDisribution = TypeDisribution.Normal,
                IsMultiIterationOptimisation = ismultiiteration,
                MaxAttempts = maxattepts,
                Tolerance = tol,
                MaxIteration = maxiteration,
                Seed = seed
            };
        }
        public static MMKConfiguration Exponential
            (
            bool ismultiiteration = true,
            int maxattepts = 100,
            int maxiteration = 1000,
            double tol = 1e-7,
            int? seed = null
            ) 
        {
            return new MMKConfiguration()
            {
                Functions = new ExponentialMMKDistribution(),
                Oprimizator = new NelderMeadOptimizator(),
                TypeDisribution = TypeDisribution.Exponential,
                IsMultiIterationOptimisation = ismultiiteration,
                MaxAttempts = maxattepts,
                Tolerance = tol,
                MaxIteration= maxiteration,
                Seed = seed
            };
        }

        public static MMKConfiguration Laplace
            (
            bool ismultiiteration = true,
            int maxattepts = 100,
            int maxiteration = 1000,
            double tol = 1e-7,
            int? seed = null
            )
        {
            return new MMKConfiguration()
            {
                Functions = new LaplaceMMKDistribution(),
                Oprimizator = new NelderMeadOptimizator(),
                TypeDisribution = TypeDisribution.Laplace,
                IsMultiIterationOptimisation = ismultiiteration,
                MaxAttempts = maxattepts,
                Tolerance = tol,
                MaxIteration = maxiteration,
                Seed = seed
            };
        }
        public static MMKConfiguration Cauchy
            (
                bool ismultiiteration = true,
                int maxattepts = 100,
                int maxiteration = 1000,
                double tol = 1e-7,
                int? seed = null
            )
        {
            return new MMKConfiguration()
            {
                Functions = new CauchyMMKDistribution(),
                Oprimizator = new NelderMeadOptimizator(),
                TypeDisribution = TypeDisribution.Cauchy,
                IsMultiIterationOptimisation = ismultiiteration,
                MaxAttempts = maxattepts,
                Tolerance = tol,
                MaxIteration = maxiteration,
                Seed = seed
            };
        }

        public static MMKConfiguration Gamma
            (
                bool ismultiiteration = true,
                int maxattepts = 100,
                int maxiteration = 1000,
                double tol = 1e-7,
                int? seed = null
            )
        {
            return new MMKConfiguration()
            {
                Functions = new GammaMMKDistribution(),
                Oprimizator = new NelderMeadOptimizator(),
                TypeDisribution = TypeDisribution.Gamma,
                IsMultiIterationOptimisation = ismultiiteration,
                MaxAttempts = maxattepts,
                Tolerance = tol,
                MaxIteration = maxiteration,
                Seed = seed
            };
        }
        public static MMKConfiguration Uniform
            (
                bool ismultiiteration = true,
                int maxattepts = 100,
                int maxiteration = 1000,
                double tol = 1e-7,
                int? seed = null
            )
        {
            return new MMKConfiguration()
            {
                Functions = new UniformMMKDistribution(),
                Oprimizator = new NelderMeadOptimizator(),
                TypeDisribution = TypeDisribution.Uniform,
                IsMultiIterationOptimisation = ismultiiteration,
                MaxAttempts = maxattepts,
                Tolerance = tol,
                MaxIteration = maxiteration,
                Seed = seed
            };
        }
    }

    public static class Test {
        public static void Main(string[] args) {
            string h = "H0";
            var n = 20;
            var thetaH0 = new Vectors([5, 0, 0, 0]);
            var valueH1 = 1.0 / (n * double.Log(n) * double.Log(n)) ;
            var thetaH1 = new Vectors([5, valueH1, valueH1, valueH1]);
            var paramDistribution = new Vectors([0, 30]);
            int numparam = 0;
            Vectors planX = new Vectors([[-1, -1, -1], 
                                        [1, -1, -1],
                                        [-1, 1, -1],
                                        [1, 1, -1],
                                        [-1, -1, 1],
                                        [1, -1, 1],
                                        [-1, 1, 1],
                                        [1, 1, 1]]);
            Vectors planP = Vectors.Ones((1,8)) / 8;
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
                        evolution: new MNKEstimator(),
                        countIteration: 10000,
                        countObservations: n,
                        numberParametr: numparam,
                        errorDist: new NormalDistribution(),
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
                statistic.Statistics.SaveToDAT(FormattableString.Invariant($"/home/zodiac/RA/ОР/Samples/H0_Parameters_{numparam}_MNKNormal{n}.dat"), title: "H0 " + statistic.ToString());
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
        }
    }
}
