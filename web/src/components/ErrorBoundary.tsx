import { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('ErrorBoundary caught error:', error, errorInfo);
    this.setState({ errorInfo });
  }

  private handleReset = (): void => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  public render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div className="bg-slate-900 border border-rose-500/40 rounded-lg p-6 my-4 shadow-xl">
          <div className="flex items-center gap-3 text-rose-400">
            <AlertTriangle className="w-6 h-6 flex-shrink-0" />
            <h2 className="text-lg font-bold text-white">
              {this.props.fallbackTitle || 'Component Rendering Error'}
            </h2>
          </div>
          <p className="text-xs text-slate-300 mt-2 font-mono bg-slate-950 p-3 rounded border border-slate-800 break-words">
            {this.state.error?.message || 'An unexpected runtime error occurred.'}
          </p>
          {this.state.errorInfo?.componentStack && (
            <details className="mt-3">
              <summary className="text-[11px] font-mono text-slate-500 cursor-pointer hover:text-slate-400">
                Component Stack Trace
              </summary>
              <pre className="text-[10px] font-mono text-slate-400 bg-slate-950 p-3 rounded mt-1 overflow-x-auto border border-slate-900 max-h-48">
                {this.state.errorInfo.componentStack}
              </pre>
            </details>
          )}
          <button
            onClick={this.handleReset}
            className="mt-4 flex items-center gap-2 px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold rounded shadow transition"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Recover & Re-render View
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
