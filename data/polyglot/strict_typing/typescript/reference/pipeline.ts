export class Pipeline<T> {
    constructor(private readonly data: ReadonlyArray<T>) {}

    map<U>(fn: (item: T) => U): Pipeline<U> {
        return new Pipeline<U>(this.data.map(fn));
    }

    filter(predicate: (item: T) => boolean): Pipeline<T> {
        return new Pipeline<T>(this.data.filter(predicate));
    }

    execute(): T[] {
        return [...this.data];
    }
}
