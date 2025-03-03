using System.Numerics;
using System.Security.Cryptography.X509Certificates;

using Regression_analysis.Regretion;
using Regression_analysis.Types;

namespace Regression_analysis
{
    
    public static class MNK
    {
        public static Vectors EstimateParametrs(IModel model, Vectors X, Vectors Y, Vectors? MatrixX = null) {

            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(X);
            ArgumentNullException.ThrowIfNull(Y);

            MatrixX ??= model.CreateMatrixX(X);

            Vectors estimatedTheta;
            try
            {
                var tMatrixX = MatrixX.T();
                EstimatedTheta = (Vectors.Inv(tMatrixX & MatrixX) & tMatrixX) & Y.T();
            }
            catch {
                throw new Exception("Error. Incorrect Data.");
            }
            return EstimatedTheta;
        }
    }

    public static class MMK
    {
        
        public static Vectors EstimateParametrsNormalDist(IModel model, Vectors X, Vectors Y, Vectors? MatrixX = null) {
            return MNK.EstimateParametrs(model, X, Y, MatrixX);
        }

        private static Vectors CheckArguments(Vectors parametr) {
            if (parametr.IsVector()) {
                if (parametr.Shape.Item2 < parametr.Shape.Item1)
                    parametr = parametr.T();
                if (parametr.Shape.Item2 != 1)
                    throw new ArgumentException("params имеет не тот размер");
                } 
            else {
                throw new ArgumentException("params не является вектором");
            }        
            return parametr;
        } 
        
        public static double LogLikelihoodChoshi(IModel model, Vectors @params, Vectors x, Vectors y) {
            ArgumentNullException.ThrowIfNull(model);
            
            var residuals = Vectors.InitVectors(y.Shape);

            x = CheckArguments(x);
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            
            Vectors funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                residuals[i] = Math.Pow(y[i] - (funcVector & @params)[0] - 1, 2);
            }
            residuals = residuals * 0.5 + 1;
            for (var i = 0; i < residuals.Size; i++)
                residuals[0, i] = Math.Log(residuals[0, i]);
            return -Vectors.Sum(residuals);
        }

        public static double LogLikelihoodExponential(IModel model, Vectors @params, params Vectors[] parametrs) {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");
            
            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            x = CheckArguments(x);
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            @paramsDist = CheckArguments(@paramsDist);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (@paramsDist.Size != 1)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {@paramsDist.Size}, а нужно 1");

            var residuals = Vectors.InitVectors(y.Shape);

            Vectors funcVector;
            for (var i = 0; i < y.Size; i++){
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                residuals[i] = y[i] - (funcVector & @params)[0];
            }
            return -@paramsDist[0] * Vectors.Sum(residuals);
        }
        public static double LogLikelihoodLaplace(IModel model, Vectors @params, params Vectors[] parametrs) {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");
            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            x = CheckArguments(x);
            y = CheckArguments(y);
            @paramsDist = CheckArguments(@paramsDist);
            @params = CheckArguments(@params);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (@paramsDist.Size != 2)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {@paramsDist.Size}, а нужно 2");

            var residuals = Vectors.InitVectors(y.Shape);

            Vectors funcVector;
            for (var i = 0; i < residuals.Size; i++) {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                residuals[i] = Math.Abs(y[i] - (funcVector & @params)[0] - @paramsDist[1]);
            }
            
            return -@paramsDist[0] * Vectors.Sum(residuals);
        }
    }

    public static class Test {
        public static void Main(string[] args) {
            Vectors test = new Vectors([1, 2, 3, 4]);
            Console.WriteLine(test.Shape);
        }
    }

}
