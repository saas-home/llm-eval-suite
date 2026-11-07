export interface OrderRecord { id?: string; qty?: number; price?: number; discount?: number; }
export interface ProcessedOrder { id: string; total: number; }
export function processOrderData(records: (OrderRecord | null | undefined)[] | null | undefined): ProcessedOrder[] {
    const res: ProcessedOrder[] = [];
    if (records) {
        for (const r of records) {
            if (r && r.qty !== undefined && r.price !== undefined) {
                if (r.qty > 0 && r.price >= 0) {
                    let total = r.qty * r.price;
                    if (r.discount !== undefined && r.discount > 0 && r.discount <= 1) {
                        total *= (1 - r.discount);
                    }
                    res.push({ id: r.id || 'N/A', total: Math.round(total * 100) / 100 });
                }
            }
        }
    }
    return res;
}
