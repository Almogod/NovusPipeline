import { Component } from 'react'

/** Keeps the page usable if the 3D layer throws (e.g. a WebGL context failure). */
export class ErrorBoundary extends Component {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch(error) {
    console.warn('3D scene disabled:', error)
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}
