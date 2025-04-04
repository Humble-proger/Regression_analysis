using System.ComponentModel.Composition.Hosting;
using System.Reflection;
using System.ComponentModel.Composition;

namespace Regression_analysis
{
    public delegate double LogLikelihoodFunction(IModel model, Vectors @params, Vectors[] parametrs);
    public delegate Vectors LogLikelihoodGradient(IModel model, Vectors @params, Vectors[] parametrs);
    public delegate Vectors LogLikelihoodGessian(IModel model, Vectors @params, Vectors[] parametrs);

    [MetadataAttribute]
    [AttributeUsage(AttributeTargets.Class)]
    public class DistributionExportAttribute : ExportAttribute {
        public string Name { get; }

        public DistributionExportAttribute(string name) : base(typeof(IMMKFunction))
        {
            Name = name;
        }
    }

    public interface IMMKFunction {
        LogLikelihoodFunction LogLikelihood { get; }
        LogLikelihoodGradient? Gradient { get; }
        LogLikelihoodGessian? Gessian { get; }
    }
    
    [DistributionExport("Choshi")]
    public class ChoshiMMKDistribution : IMMKFunction {
        public static string Name => "Choshi";

        public LogLikelihoodFunction LogLikelihood => (model, theta, @params) => 
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];
            
            double residuals = 0.0;

            double funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = y[i] - (model.VectorFunc(Vectors.GetRow(x, i)) & theta)[0] - paramsDist[0];
                residuals -= Math.Log(1 + Math.Pow(funcVector / paramsDist[1], 2));
            }
            return residuals;
        };

        public LogLikelihoodGradient? Gradient => (model, theta, @params) => 
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];

            var residuals = Vectors.Zeros((1, model.CountRegressor));

            double funcVector;
            Vectors vector;
            for (var i = 0; i < y.Size; i++)
            {
                vector = model.VectorFunc(Vectors.GetRow(x, i));
                funcVector = y[i] - (vector & theta)[0] - paramsDist[0];
                residuals += (funcVector / (paramsDist[1] * paramsDist[1] + funcVector * funcVector)) * vector;
            }
            return 2 * residuals;
        };

        public LogLikelihoodGessian? Gessian => (model, theta, @params) => 
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];

            var residuals = Vectors.InitVectors((model.CountRegressor, model.CountRegressor));
            double funcVector, temp = double.Pow(paramsDist[1], 2);
            Vectors matrixM;
            for (var i = 0; i < y.Size; i++) {
                matrixM = model.VectorFunc(Vectors.GetRow(x, i));
                funcVector = double.Pow(y[i] - (matrixM & theta)[0] - paramsDist[0], 2);
                matrixM = matrixM.T() & matrixM;
                residuals += ((funcVector - temp) / (funcVector + temp)) * matrixM;
            }
            return 2 * residuals;
        };
    }

    [DistributionExport("Exponential")]
    public class ExponentialMMKDistribution : IMMKFunction
    {
        public static string Name => "Exponential";

        public LogLikelihoodFunction LogLikelihood => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];

            double residuals = 0.0;

            double funcVector;
            double penaity = 0.0;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = y[i] - (model.VectorFunc(Vectors.GetRow(x, i)) & theta)[0];
                residuals -= funcVector - paramsDist[0];
                penaity += Math.Pow(double.Abs(funcVector) - funcVector, 2) / 2;
            }
            return (1 / @paramsDist[1]) * residuals - penaity;
        };

        public LogLikelihoodGradient? Gradient => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];

            Vectors residuals = Vectors.Zeros((1, model.CountRegressor));

            Vectors funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                residuals += funcVector;
            }
            return (1 / @paramsDist[1]) * residuals;
        };

        public LogLikelihoodGessian? Gessian => null;
    }

    [DistributionExport("Laplace")]
    public class LaplaceMMKDistribution : IMMKFunction
    {
        public static string Name => "Laplace";

        public LogLikelihoodFunction LogLikelihood => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];

            double residuals = 0.0;

            Vectors funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                residuals -= double.Abs(y[i] - (funcVector & theta)[0] - paramsDist[0]);
            }

            return paramsDist[1] * residuals;
        };

        public LogLikelihoodGradient? Gradient => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];

            Vectors residuals = Vectors.Zeros((1, model.CountRegressor));

            Vectors funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                residuals -= double.Sign(y[i] - (funcVector & theta)[0] - paramsDist[0]) * funcVector;
            }

            return paramsDist[1] * residuals;
        };

        public LogLikelihoodGessian? Gessian => null;
    }

    [DistributionExport("Gamma")]
    public class GammaMMKDistribution : IMMKFunction
    {
        public static string Name => "Gamma";

        public LogLikelihoodFunction LogLikelihood => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];

            double residuals = 0.0;

            double funcVector;
            double penaity = 0.0;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = y[i] - (model.VectorFunc(Vectors.GetRow(x, i)) & theta)[0];
                residuals += (paramsDist[2] - 1) * double.Log(funcVector - paramsDist[0]) - (funcVector - paramsDist[0]) / paramsDist[1];
                penaity += double.Pow(double.Abs(funcVector) - funcVector, 2) / 2;
            }

            return residuals - penaity;
        };

        public LogLikelihoodGradient? Gradient => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];

            Vectors residuals = Vectors.Zeros((1, model.CountRegressor));

            Vectors funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                var e = y[i] - (funcVector & theta)[0] - paramsDist[0];
                residuals += (1 / paramsDist[1] - (paramsDist[2] - 1) / e) * funcVector;
            }

            return residuals;
        };

        public LogLikelihoodGessian? Gessian => (model, theta, @params) => 
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];

            Vectors residuals = Vectors.Zeros((model.CountRegressor, model.CountRegressor));
            double funcValue;
            Vectors matrixM;
            for (int i = 0; i < y.Size; i++) {
                matrixM = model.VectorFunc(Vectors.GetRow(x, i));
                funcValue = double.Pow(y[i] - (matrixM & theta)[0] - paramsDist[0], 2);
                matrixM = matrixM.T() & matrixM;
                residuals += matrixM / funcValue;
            }
            return (paramsDist[2] - 1) * residuals;
        };
    }

    [DistributionExport("Uniform")]
    public class UniformMMKDistribution : IMMKFunction
    {
        public static string Name => "Uniform";

        public LogLikelihoodFunction LogLikelihood => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];

            double maxEps = double.MinValue;
            double minEps = double.MaxValue;

            double funcVector;
            var n = y.Size;
            for (var i = 0; i < n; i++)
            {
                funcVector = y[i] - (model.VectorFunc(Vectors.GetRow(x, i)) & theta)[0];
                if (funcVector > maxEps)
                    maxEps = funcVector;
                if (funcVector < minEps)
                    minEps = funcVector;
            }
            return minEps - maxEps;
        };

        public LogLikelihoodGradient? Gradient => null;

        public LogLikelihoodGessian? Gessian => null;
    }
    [DistributionExport("Normal")]
    public class NormalMMKDistribution : IMMKFunction
    {
        public static string Name => "Normal";

        public LogLikelihoodFunction LogLikelihood => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];

            double residuals = 0.0;

            double funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = y[i] - (model.VectorFunc(Vectors.GetRow(x, i)) & theta)[0] - paramsDist[0];
                residuals -= double.Pow(funcVector, 2);
            }

            return residuals / (2 * double.Pow(paramsDist[1], 2));
        };

        public LogLikelihoodGradient? Gradient => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];

            Vectors residuals = Vectors.Zeros((1, model.CountRegressor));

            Vectors funcVector;
            double funcValue;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                funcValue = y[i] - (funcVector & theta)[0] - paramsDist[0];
                residuals += funcValue * funcVector;
            }

            return residuals / double.Pow(paramsDist[1], 2);
        };

        public LogLikelihoodGessian? Gessian => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];

            var matrixX = model.CreateMatrixX(x);
            return -(matrixX.T() & matrixX) / double.Pow(paramsDist[1], 2);
        };
    }

    public class DistributionMMKFactory {
        [ImportMany]
        private IEnumerable<Lazy<IMMKFunction, DistributionExportAttribute>> _distibutions;
        
        private readonly Dictionary<string, IMMKFunction> _loadedDistributions = [];

        public void Initialize() {
            var catalog = new AssemblyCatalog(Assembly.GetExecutingAssembly());
            var container = new CompositionContainer(catalog);
            container.ComposeParts(this);

            foreach (var export in _distibutions) {
                _loadedDistributions[export.Metadata.Name] = export.Value;
            }
        }
        public IMMKFunction? GetDistribution(string name) {
            return _loadedDistributions.TryGetValue(name, out var dist) ? dist : null;
        }
    }




    /*
    public class MMKFunctions
    {

        public static Vectors EstimateParametrsNormalDist(IModel model, Vectors x, Vectors y, Vectors? matrixX = null)
        {
            return MNK.EstimateParametrs(model, x, y, matrixX);
        }

        private static Vectors CheckArguments(Vectors parametr)
        {
            if (parametr.IsVector())
            {
                if (parametr.Shape.Item2 < parametr.Shape.Item1)
                    parametr = parametr.T();
                if (parametr.Shape.Item1 != 1)
                    throw new ArgumentException("params имеет не тот размер");
            }
            else
            {
                throw new ArgumentException("params не является вектором");
            }
            return parametr;
        }

        public static double LogLikelihoodChoshiDist(IModel model, Vectors @params, params Vectors[] parametrs)
        {
            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");

            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            @paramsDist = CheckArguments(@paramsDist);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (@paramsDist.Size != 2)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {@paramsDist.Size}, а нужно 2");

            double residuals = 0.0;

            double funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = y[i] - (model.VectorFunc(Vectors.GetRow(x, i)) & @params)[0] - @paramsDist[0];
                residuals -= Math.Log(1 + Math.Pow(funcVector / paramsDist[1], 2));
            }
            return residuals;
        }

        public static Vectors DfLogLikelihoodChoshiDist(IModel model, Vectors @params, params Vectors[] parametrs)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");

            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            paramsDist = CheckArguments(paramsDist);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (paramsDist.Size != 2)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {paramsDist.Size}, а нужно 2");

            var residuals = Vectors.Zeros((1, model.CountRegressor));

            double funcVector;
            Vectors vector;
            for (var i = 0; i < y.Size; i++)
            {
                vector = model.VectorFunc(Vectors.GetRow(x, i));
                funcVector = y[i] - (vector & @params)[0] - paramsDist[0];
                residuals += (funcVector / (paramsDist[1] * paramsDist[1] + funcVector * funcVector)) * vector;
            }
            return 2 * residuals;
        }

        public static Vectors GessianLogLikelihoodChoshiDist(IModel model, Vectors @params, params Vectors[] parametrs)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");

            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            paramsDist = CheckArguments(paramsDist);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (paramsDist.Size != 2)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {paramsDist.Size}, а нужно 2");

            var residuals = Vectors.Zeros((model.CountRegressor, model.CountRegressor));

            double funcVector;
            Vectors vector;
            for (var i = 0; i < y.Size; i++)
            {
                vector = model.VectorFunc(Vectors.GetRow(x, i));
                funcVector = Math.Pow(y[i] - (vector & @params)[0] - paramsDist[0], 2);
                var value = paramsDist[1] * paramsDist[1];
                residuals += ((funcVector - value) / (value + funcVector)) * (vector.T() & vector);
            }
            return 2 * residuals;
        }

        public static double LogLikelihoodExponentialDist(IModel model, Vectors @params, params Vectors[] parametrs)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");

            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            paramsDist = CheckArguments(paramsDist);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (paramsDist.Size != 2)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {paramsDist.Size}, а нужно 2");

            double residuals = 0.0;

            double funcVector;
            double penaity = 0.0;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = y[i] - (model.VectorFunc(Vectors.GetRow(x, i)) & @params)[0];
                residuals -= funcVector - paramsDist[0];
                penaity += Math.Pow(double.Abs(funcVector) - funcVector, 2) / 2;
            }
            return (1 / @paramsDist[1]) * residuals - penaity;
        }

        public static Vectors DfLogLikelihoodExponentialDist(IModel model, Vectors @params, params Vectors[] parametrs)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");

            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            paramsDist = CheckArguments(paramsDist);
            if (paramsDist.Size != 2)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {paramsDist.Size}, а нужно 2");

            Vectors residuals = Vectors.Zeros((1, model.CountRegressor));

            Vectors funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                residuals += funcVector;
            }
            return (1 / @paramsDist[1]) * residuals;
        }
        public static double LogLikelihoodLaplaceDist(IModel model, Vectors @params, params Vectors[] parametrs)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");
            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            paramsDist = CheckArguments(paramsDist);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (paramsDist.Size != 2)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {paramsDist.Size}, а нужно 2");

            double residuals = 0.0;

            Vectors funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                residuals -= Math.Abs(y[i] - (funcVector & @params)[0] - paramsDist[0]);
            }

            return paramsDist[1] * residuals;
        }

        public static Vectors DfLogLikelihoodLaplaceDist(IModel model, Vectors @params, params Vectors[] parametrs)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");
            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            paramsDist = CheckArguments(paramsDist);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (paramsDist.Size != 2)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {paramsDist.Size}, а нужно 2");

            Vectors residuals = Vectors.Zeros((1, model.CountRegressor));

            Vectors funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                residuals -= Math.Sign(y[i] - (funcVector & @params)[0] - paramsDist[0]) * funcVector;
            }

            return paramsDist[1] * residuals;
        }

        public static double LogLikelihoodGammaDist(IModel model, Vectors @params, params Vectors[] parametrs)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");
            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            paramsDist = CheckArguments(paramsDist);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (paramsDist.Size != 3)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {paramsDist.Size}, а нужно 3");

            double residuals = 0.0;

            double funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = y[i] - (model.VectorFunc(Vectors.GetRow(x, i)) & @params)[0] - paramsDist[0];
                residuals += (@paramsDist[2] - 1) * Math.Log(funcVector) - funcVector / paramsDist[1];
            }

            return residuals;
        }

        public static Vectors DfLogLikelihoodGammaDist(IModel model, Vectors @params, params Vectors[] parametrs)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");
            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            paramsDist = CheckArguments(paramsDist);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (paramsDist.Size != 3)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {paramsDist.Size}, а нужно 3");

            Vectors residuals = Vectors.Zeros((1, model.CountRegressor));

            Vectors funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                var e = y[i] - (funcVector & @params)[0] - paramsDist[0];
                residuals += (1 / paramsDist[1] - (paramsDist[2] - 1) / e) * funcVector;
            }

            return residuals;
        }
        public static double LogLikelihoodUniformDist(IModel model, Vectors @params, params Vectors[] parametrs)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 2)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 2.");
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();

            double maxEps = double.MinValue;
            double minEps = double.MaxValue;

            double funcVector;
            var n = y.Size;
            for (var i = 0; i < n; i++)
            {
                funcVector = y[i] - (model.VectorFunc(Vectors.GetRow(x, i)) & @params)[0];
                if (funcVector > maxEps)
                    maxEps = funcVector;
                if (funcVector < minEps)
                    minEps = funcVector;
            }
            return minEps - maxEps;
        }
    }
    */
}
