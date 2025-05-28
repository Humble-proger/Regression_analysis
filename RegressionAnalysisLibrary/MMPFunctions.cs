namespace RegressionAnalysisLibrary
{
    public delegate double LogLikelihoodFunction(IModel model, Vectors @params, Vectors[] parametrs);
    public delegate Vectors LogLikelihoodGradient(IModel model, Vectors @params, Vectors[] parametrs);
    public delegate Vectors LogLikelihoodGessian(IModel model, Vectors @params, Vectors[] parametrs);
    public delegate double? Moment(Vectors paramDist);


    public interface IMMPFunction
    {
        string Name { get; }
        LogLikelihoodFunction LogLikelihood { get; }
        LogLikelihoodGradient? Gradient { get; }
        LogLikelihoodGessian? Gessian { get; }
    }

    public class MMPConfiguration
    {

        public required IMMPFunction Functions { get; set; }
        public required IOprimizator Oprimizator { get; set; }
        public required Moment Mean { get; set; }
        public bool IsMultiIterationOptimisation { get; set; } = true;
        public int MaxAttempts { get; set; } = 100;
        public double Tolerance { get; set; } = 1e-7;
        public int MaxIteration { get; set; } = 1000;
        public bool MNKEstuminate { get; set; } = true;
        public int? Seed { get; set; }
    }

    public class CauchyMMPDistribution : IMMPFunction
    {
        public string Name => "Cauchy";

        public LogLikelihoodFunction LogLikelihood => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];
            theta = theta.T();

            var residuals = 0.0;
            var scaleSq = paramsDist[1] * paramsDist[1];
            double funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = y[i] - (model.VectorFunc(Vectors.GetRow(x, i)) & theta)[0] - paramsDist[0];
                residuals += double.Log(scaleSq + funcVector * funcVector);
            }
            return residuals;
        };

        public LogLikelihoodGradient? Gradient => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];

            var residuals = Vectors.Zeros((1, model.CountRegressor));
            theta = theta.T();
            double funcVector;
            var scaleSq = paramsDist[1] * paramsDist[1];
            double denominator;
            Vectors vector;
            for (var i = 0; i < y.Size; i++)
            {
                vector = model.VectorFunc(Vectors.GetRow(x, i));
                funcVector = y[i] - (vector & theta)[0] - paramsDist[0];
                denominator = scaleSq + funcVector * funcVector;
                residuals += funcVector / denominator * vector;
            }
            return -2 * residuals;
        };

        public LogLikelihoodGessian? Gessian => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];

            theta = theta.T();
            var residuals = Vectors.InitVectors((model.CountRegressor, model.CountRegressor));
            double funcVector, temp = double.Pow(paramsDist[1], 2);
            Vectors matrixM;
            for (var i = 0; i < y.Size; i++)
            {
                matrixM = model.VectorFunc(Vectors.GetRow(x, i));
                funcVector = double.Pow(y[i] - (matrixM & theta)[0] - paramsDist[0], 2);
                matrixM = matrixM.T() & matrixM;
                residuals += (funcVector - temp) / double.Pow(funcVector + temp, 2) * matrixM;
            }
            return -2 * residuals;
        };
    }

    public class ExponentialMMPDistribution : IMMPFunction
    {
        public string Name => "Exponential";

        public LogLikelihoodFunction LogLikelihood => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];
            theta = theta.T();
            var residuals = 0.0;

            double funcVector;
            var penaity = 0.0;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = y[i] - (model.VectorFunc(Vectors.GetRow(x, i)) & theta)[0] - paramsDist[0];
                residuals += funcVector;
                penaity += double.Max(0, -funcVector) * 1e+6;
            }
            return residuals / @paramsDist[1] + penaity;
        };

        public LogLikelihoodGradient? Gradient => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];
            theta = theta.T();

            var residuals = Vectors.Zeros((1, model.CountRegressor));

            Vectors funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                residuals += funcVector;
            }
            return 1 / @paramsDist[1] * residuals;
        };

        public LogLikelihoodGessian? Gessian => null;
    }

    public class LaplaceMMPDistribution : IMMPFunction
    {
        public string Name => "Laplace";

        public LogLikelihoodFunction LogLikelihood => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];
            theta = theta.T();
            var residuals = 0.0;

            for (var i = 0; i < y.Size; i++)
            {
                var funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                residuals += double.Abs(y[i] - (funcVector & theta)[0] - paramsDist[0]);
            }

            return residuals;
        };

        public LogLikelihoodGradient? Gradient => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];
            theta = theta.T();

            var residuals = Vectors.Zeros((1, model.CountRegressor));

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

    public class GammaMMPDistribution : IMMPFunction
    {
        public string Name => "Gamma";

        public LogLikelihoodFunction LogLikelihood => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];
            theta = theta.T();

            var residuals = 0.0;

            double funcVector;
            var penaity = 0.0;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = y[i] - (model.VectorFunc(Vectors.GetRow(x, i)) & theta)[0] - paramsDist[0];
                residuals -= (paramsDist[2] - 1) * double.Log(double.Max(1e-14, funcVector)) - funcVector / paramsDist[1];
                penaity += double.Max(0, -funcVector) * 1e+6;
            }

            return residuals + penaity;
        };

        public LogLikelihoodGradient? Gradient => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];
            theta = theta.T();

            var residuals = Vectors.Zeros((1, model.CountRegressor));

            Vectors funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                var e = y[i] - (funcVector & theta)[0] - paramsDist[0];
                residuals += (1 / paramsDist[1] - (paramsDist[2] - 1) / e) * funcVector;
            }

            return -residuals;
        };

        public LogLikelihoodGessian? Gessian => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];
            theta = theta.T();

            var residuals = Vectors.Zeros((model.CountRegressor, model.CountRegressor));
            double funcValue;
            Vectors matrixM;
            for (var i = 0; i < y.Size; i++)
            {
                matrixM = model.VectorFunc(Vectors.GetRow(x, i));
                funcValue = double.Pow(y[i] - (matrixM & theta)[0] - paramsDist[0], 2);
                matrixM = matrixM.T() & matrixM;
                residuals += matrixM / funcValue;
            }
            return (paramsDist[2] - 1) * residuals;
        };
    }

    public class UniformMMPDistribution : IMMPFunction
    {
        public string Name => "Uniform";

        public LogLikelihoodFunction LogLikelihood => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];
            theta = theta.T();

            var maxEps = double.MinValue;
            var minEps = double.MaxValue;

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
            return double.Log(maxEps - minEps);
        };

        public LogLikelihoodGradient? Gradient => null;

        public LogLikelihoodGessian? Gessian => null;
    }
    public class NormalMMPDistribution : IMMPFunction
    {
        public string Name => "Normal";

        public LogLikelihoodFunction LogLikelihood => (model, theta, @params) =>
        {
            var paramsDist = @params[0];
            var x = @params[1];
            var y = @params[2];
            theta = theta.T();

            var residuals = 0.0;

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
            theta = theta.T();

            var residuals = Vectors.Zeros((1, model.CountRegressor));

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
}
