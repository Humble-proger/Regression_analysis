using System.Numerics;
using System.Security.Cryptography.X509Certificates;

using Regression_analysis.Regretion;
using Regression_analysis.Types;

namespace Regression_analysis.Evolution
{
    
    public static class MNK
    {
        public static Vectors EstimateParametrs(IModel model, Vectors X, Vectors Y, Vectors? MatrixX = null) {

            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(X);
            ArgumentNullException.ThrowIfNull(Y);

            if (MatrixX is null)
                MatrixX = model.CreateMatrixX(X);

            Vectors EstimatedTheta;
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

        public static double LogLikelihoodChoshi(IModel model, Vectors @params, Vectors x, Vectors y) {
            var residuals = Vectors.InitVectors(y.Shape);
            @params = @params.T();
             
            Vectors funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                residuals[i] = Math.Pow(y[i] - (funcVector & @params)[0] - 1, 2);
            }
            residuals = residuals * 0.5 + 1;
            for (var i = 0; i < residuals.Shape.Item1; i++)
                for (var j = 0; j < residuals.Shape.Item2; j++)
                    residuals[i, j] = Math.Log(residuals[i, j]);
            return Vectors.Sum(residuals);
        }
    }
}
