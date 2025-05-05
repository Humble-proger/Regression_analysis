using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;

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
                Vectors initParams;
                try {
                    var matrixX = model.CreateMatrixX(otherParameters[1]);
                    var tMatrixX = matrixX.T();
                    var mean = Config.Mean(otherParameters[0]);
                    initParams =  mean is null 
                        ? (Vectors.Inv(tMatrixX & matrixX) & tMatrixX) & otherParameters[2].T()
                        : (Vectors.Inv(tMatrixX & matrixX) & tMatrixX) & (otherParameters[2] - (double) mean).T();
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
                        (1, model.CountRegressor),
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
                Mean = NormalDistribution.Mean,
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
                Mean = ExponentialDistribution.Mean,
                IsMultiIterationOptimisation = ismultiiteration,
                MaxAttempts = maxattepts,
                Tolerance = tol,
                MaxIteration= maxiteration,
                Seed = seed,
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
                Mean = LaplaceDistribution.Mean,
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
                Mean = CauchyDistribution.Mean,
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
                Mean = GammaDistribution.Mean,
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
                Mean = UniformDistribution.Mean,
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
            var n = 1000;
            var thetaH0 = new Vectors([5, 2, 3, 7]);
            var valueH1 = 1.0 / (n * double.Log(n) * double.Log(n)) ;
            var thetaH1 = new Vectors([5, valueH1, valueH1, valueH1]);
            var paramDistribution = new Vectors([0, 30]);
            int numparam = 1;

            //var e = LaplaceDistribution.Generate((1, 10000), paramDistribution, new Random());
            //e.SaveToDAT($"D:/Program/Budancev/ОР/Samples/Laplace.dat", "Laplace (0, 30)");
            
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
                            thetaH0,
                            true
                        );

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
            
        }
    }
}
