import { Component } from '@angular/core';
import { NgIf } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../api-service';

@Component({
  selector: 'app-mbti',
  standalone: true,
  imports: [NgIf, FormsModule],
  templateUrl: './mbti.component.html',
  styleUrls: ['./mbti.component.css']
})
export class MbtiComponent {
  inputText = '';
  result: string | null = null;
  loading = false;

  constructor(private apiService: ApiService) {}

  onSubmit(): void {
  if (!this.inputText.trim()) return;

  // 🔧 防呆處理：將特殊雙引號/單引號替換成標準引號
  const cleanedText = this.inputText
    .replace(/[“”]/g, '"')    // 中文雙引號 → 英文雙引號
    .replace(/[‘’]/g, "'");   // 中文單引號 → 英文單引號

  this.loading = true;
  this.result = null;

  this.apiService.predictMBTI(cleanedText).subscribe({
    next: (res) => {
      this.result = `${res.IE}${res.SN}${res.TF}${res.JP}`;
      this.loading = false;
    },
    error: (err) => {
      alert('伺服器錯誤：' + JSON.stringify(err.error));
      this.loading = false;
    }
  });
}
}

