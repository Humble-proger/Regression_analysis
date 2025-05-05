using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using CommunityToolkit.Mvvm;
using CommunityToolkit.Mvvm.ComponentModel;

using Regression_analysis;


namespace RegressionAnalysisApplication
{

    public partial class MainWindowViewModel : ObservableObject
    {
        public ObservableCollection<Distribution> ErrorDistibution { get; set; } = [];

        [ObservableProperty]
        private Distribution? _selectDistribution;

        public MainWindowViewModel() {
            LoadErrorDistribuions();
            if (ErrorDistibution.Count > 0)
                SelectDistribution = ErrorDistibution[0];

        }

        private void LoadErrorDistribuions() {
            var result = new List<Distribution>();
            var fabric = new DistributionFactory();
            Vectors defaultParam;
            string[] nameParam;
            (double?, double?)[]? bounds;
            FactorBound? factorBound;
            foreach (var elem in fabric.GetAllDistributions()) {
                var parameters = new List<Parameter>();
                defaultParam = elem.Instance.DefaultParametrs;
                nameParam = elem.Instance.NameParameters;
                bounds = elem.Instance.BoundsParameters;
                if (bounds is not null) {
                    for (int i = 0; i < elem.Instance.CountParametrsDistribution; i++) {
                        factorBound = new FactorBound(bounds[i].Item1, bounds[i].Item2);
                        parameters.Add(new Parameter(nameParam[i], defaultParam[i], factorBound));
                    }
                }
                else {
                    factorBound = null;
                    for (int i = 0; i < elem.Instance.CountParametrsDistribution; i++)
                    {
                        parameters.Add(new Parameter(nameParam[i], defaultParam[i], factorBound));
                    }
                }
                result.Add(new Distribution(elem.Name, parameters, elem.Instance));
            }
            ErrorDistibution = new ObservableCollection<Distribution>(result);
        }
    }
    public class ViewModelBase : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;

        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        protected bool SetField<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
        {
            if (EqualityComparer<T>.Default.Equals(field, value))
                return false;

            field = value;
            OnPropertyChanged(propertyName);
            return true;
        }
    }

    public class Distribution : ObservableObject
    {
        public string Name { get; }
        public IRandomDistribution RandomDistribution { get; }
        public ObservableCollection<Parameter> Parameters { get; }

        public Distribution(string name, IEnumerable<Parameter> parameters, IRandomDistribution randomDistribution)
        {
            Name = name;
            Parameters = new ObservableCollection<Parameter>(parameters);
            RandomDistribution = randomDistribution;
        }
    }
    public class Parameter : ObservableObject
    {
        public string Name { get; }

        private double _value;

        public double Value 
        { 
            get => _value; 
            set {
                if (Bounds is not null)
                {
                    if (Bounds.Min is not null && value < Bounds.Min)
                        return;
                    else if (Bounds.Max is not null && value > Bounds.Max)
                        return;
                    else
                        _value = value;
                }
                else {
                    _value = value;
                }
                OnPropertyChanged(nameof(Value));
            } 
        }

        public FactorBound? Bounds { get; }

        public Parameter(string name, double value, FactorBound? bound)
        {
            Name = name;
            Value = value;
            Bounds = bound;
        }
    }

    public class FactorBound(double? min, double? max)
    {
        public double? Min { get; set; } = min;
        public double? Max { get; set; } = max;
    }
}
