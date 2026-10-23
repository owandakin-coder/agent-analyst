// Offline report from scoped observations. Never submits orders or changes models.
const fs = require('node:fs');
const path = require('node:path');
const {summarize} = require('../workspace-evidence.js');
function markdown(report) {
  const pct = value => (value*100).toFixed(4)+'%';
  return `# ATZMA — דוח איכות האסטרטגיה

תוצאת הבדיקה: **אין עדיין ראיות מספיקות להוכחת רווחיות או יתרון מתמשך.**

תקופה משותפת: ${report.first_day || 'אין נתונים'} עד ${report.last_day || 'אין נתונים'}.
${report.paired_cycles} מחזורי מחקר מזווגים, ${report.daily_marks} נקודות יומיות ו־${report.return_intervals} מרווחי תשואה.
חסרים לפחות ${report.remaining_intervals} מרווחים לסף הזמן המינימלי. הסף לבדו אינו מאשר מודל; למועמד חדש נדרש חלון משלו לאחר הצבתו.

| תיק מחקר מדומה | תשואה בתקופה המשותפת | ירידה מרבית במדידות היומיות |
|---|---:|---:|
${report.comparisons.map(row=>`| ${row.strategy} | ${pct(row.return)} | ${pct(row.max_drawdown)} |`).join('\n')}

כל התיקים נורמלו לנקודת ההתחלה המשותפת; אלה אינן תשואות מאז הקמת כל תיק.
תקרת החשיפה היא 50% והעלויות המשוערות הן 10 נקודות בסיס לכל מחזור כספי. מדד ההשוואה הוא 50% SPY ו־50% מזומן.
המדידות הן התצפיות המשותפות האחרונות בכל יום לפני היום הנוכחי, ולא בהכרח שערי הסגירה.

## פירוש התוצאה

${report.champion_exposure_observations && !report.champion_invested_cycles ? `תיק המחקר של המודל נשאר ללא חשיפה בכל ${report.champion_exposure_observations} המחזורים שנמדדו. תשואה אפסית משקפת מזומן ואינה הוכחה לאיכות האסטרטגיה.` : 'תשואה אפסית במודל אינה הוכחה לאיכות: היא עשויה לנבוע מהיעדר חשיפה. יש לבדוק אותה מול יומן ההחלטות והחזקות המחקר.'}
בדיקות תקינות, מחזורי polling ועסקאות שאושרו על ידי הברוקר אינם תחליף להוכחת ביצועים.
יתרת חשבון Paper עשויה לכלול הפקדות ופעולות ידניות; היא אינה תשואת אסטרטגיה מנוטרלת תזרימים. עמלות בפועל אינן מאומתות.
אין כאן Sharpe שנתי לתקופה קצרה או הצהרה על מובהקות סטטיסטית.

## תנאים שנותרו

- להמשיך איסוף בימי מסחר בפועל ולבדוק את התוצאות באותו חלון השוואה.
- מועמד חדש חייב לעבור סינון היסטורי ולהיצבר עבורו לפחות 21 נקודות יומיות מזווגות לאחר הצבתו.
- תהליך האישור הנפרד בודק XNYS, חגים וסגירות מוקדמות, התאמת מודל/חשבון ועלויות, ותנאי תשואה וסיכון.
- לפני הרחבת רשימת המניות: לאמת עומק היסטורי, פיצולים ודיבידנדים ופיזור ענפי, ולאמן ולבדוק מודל מתאים.

חלון נתונים מוגבל: ${report.truncated ? 'כן' : 'לא'}. קבוצות לא שלמות שהושמטו: ${report.incomplete_groups}.
`;
}
if (require.main === module) {
  const [input, output] = process.argv.slice(2);
  if (!input || !output) throw Error('Usage: node scripts/evidence_report.cjs scoped-observations.json report.md');
  const rows = JSON.parse(fs.readFileSync(input,'utf8'));
  const scopes = new Set(rows.map(r=>JSON.stringify([r.user_id || '', r.broker_account_id || ''])));
  if (scopes.size > 1) throw Error('Export one account at a time; reports cannot pool accounts.');
  const report = summarize(rows);
  fs.mkdirSync(path.dirname(output),{recursive:true}); fs.writeFileSync(output,markdown(report));
  console.log(JSON.stringify({daily_marks:report.daily_marks,return_intervals:report.return_intervals,profitability_proven:false}));
}
module.exports = {markdown};
