export interface OrderRecord { id?: string; qty?: number; price?: number; discount?: number; }
export interface ProcessedOrder { id: string; total: number; }

const hasValidDiscount = (d: number | undefined): d is number => d !== undefined && d > 0 && d <= 1;

export function processOrderData(records: (OrderRecord | null | undefined)[] | null | undefined): ProcessedOrder[] {
    if (!records) return [];
    const res: ProcessedOrder[] = [];
    for (const r of records) {
        if (!r || r.qty === undefined || r.price === undefined) continue;
        if (r.qty <= 0 || r.price < 0) continue;
        let total = r.qty * r.price;
        if (hasValidDiscount(r.discount)) total *= (1 - r.discount);
        res.push({ id: r.id || 'N/A', total: Math.round(total * 100) / 100 });
    }
    return res;
}
