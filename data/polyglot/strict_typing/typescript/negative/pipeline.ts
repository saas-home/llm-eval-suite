export class Pipeline<T> {
    constructor(private readonly data: ReadonlyArray<T>) {}

    map(fn: (item: T) => any): Pipeline<any> {
        return new Pipeline(this.data.map(fn));
    }

    filter(predicate: (item: T) => boolean): Pipeline<T> {
        return new Pipeline<T>(this.data.filter(predicate));
    }

    execute(): T[] {
        return [...this.data];
    }
}
